# Build-time input capture, not configure-time Git sampling. The always-run
# preparation target uses content-stable writes; only changed metadata rebuilds
# generate.cpp. The publication target also runs on a no-op, validating the
# actual linked bytes before creating/updating the adjacent receipt.
function(strata_build_provenance target root backend)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(_owner "${root}/tools/build_provenance.py")
  set(_stage "${CMAKE_CURRENT_BINARY_DIR}/provenance/$<CONFIG>")
  set(_cuda "")
  if(backend STREQUAL "cuda")
    set(_cuda "${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
  set(_arch "${CMAKE_CUDA_ARCHITECTURES}${CMAKE_HIP_ARCHITECTURES}${STRATA_SYCL_AOT}")
  # Hash configuration privately; never embed machine paths or raw flags. Limit
  # captured cache entries to native build options, not arbitrary caller data.
  get_cmake_property(_cache CACHE_VARIABLES)
  list(SORT _cache)
  set(_flags "")
  foreach(_key IN LISTS _cache)
    if(_key MATCHES "^(CMAKE_(C|CXX|CUDA|HIP)_FLAGS|CMAKE_EXE_LINKER_FLAGS|CMAKE_MSVC_RUNTIME_LIBRARY|STRATA_(ENABLE_|NATIVE_|PORTABLE|ISA_FLOOR|PREFILL_|MMQ_|Q6K_|ORCA_|EXPERIMENTAL_|HIP_|SYCL_)|GGML_)")
      string(APPEND _flags "${_key}=${${_key}}\n")
    endif()
  endforeach()
  set(_content "{\n")
  foreach(_pair IN ITEMS version backend configuration architectures cuda cxx c_compiler cuda_compiler hip_compiler toolchain_file ggml flags generator)
    if(_pair STREQUAL "version")
      set(_value "${PROJECT_VERSION}")
    elseif(_pair STREQUAL "backend")
      set(_value "${backend}")
    elseif(_pair STREQUAL "configuration")
      set(_value "$<CONFIG>")
    elseif(_pair STREQUAL "architectures")
      set(_value "${_arch}")
    elseif(_pair STREQUAL "cuda")
      set(_value "${_cuda}")
    elseif(_pair STREQUAL "cxx")
      set(_value "${CMAKE_CXX_COMPILER}")
    elseif(_pair STREQUAL "c_compiler")
      set(_value "${CMAKE_C_COMPILER}")
    elseif(_pair STREQUAL "toolchain_file")
      set(_value "${CMAKE_TOOLCHAIN_FILE}")
    elseif(_pair STREQUAL "cuda_compiler")
      set(_value "${CMAKE_CUDA_COMPILER}")
    elseif(_pair STREQUAL "hip_compiler")
      set(_value "${CMAKE_HIP_COMPILER}")
    elseif(_pair STREQUAL "ggml")
      set(_value "${STRATA_GGML_SOURCE_DIR}")
    elseif(_pair STREQUAL "flags")
      set(_value "${_flags}\ncxx=${CMAKE_CXX_COMPILER_ID}:${CMAKE_CXX_COMPILER_VERSION}")
    else()
      set(_value "${CMAKE_GENERATOR}")
    endif()
    string(REPLACE "\\" "\\\\" _value "${_value}")
    string(REPLACE "\"" "\\\"" _value "${_value}")
    string(REPLACE "\n" "\\n" _value "${_value}")
    string(REPLACE "\r" "\\r" _value "${_value}")
    string(REPLACE "\t" "\\t" _value "${_value}")
    string(APPEND _content "\"${_pair}\":\"${_value}\",\n")
  endforeach()
  string(APPEND _content "\"owner\":1,\"target_options\":\"target-options.txt\"}\n")
  file(GENERATE OUTPUT "${_stage}/inputs.json" CONTENT "${_content}"
    CONDITION "$<COMPILE_LANGUAGE:CXX>" TARGET ${target})
  # Evaluated definitions can contain quotes/backslashes; hash their exact text
  # separately instead of inserting generator-expression results into JSON.
  file(GENERATE OUTPUT "${_stage}/target-options.txt"
    CONTENT "link=$<TARGET_PROPERTY:${target},LINK_OPTIONS>\ncompile=$<TARGET_PROPERTY:${target},COMPILE_OPTIONS>\ndefines=$<TARGET_PROPERTY:${target},COMPILE_DEFINITIONS>\n"
    CONDITION "$<COMPILE_LANGUAGE:CXX>" TARGET ${target})
  file(GENERATE OUTPUT "${_stage}/executable.txt" CONTENT "$<TARGET_FILE:${target}>")
  add_custom_target(${target}_build_inputs
    COMMAND "${Python3_EXECUTABLE}" "${_owner}" prepare --root "${root}" --inputs "${_stage}/inputs.json" --stage "${_stage}"
    BYPRODUCTS "${_stage}/strata/build_info.hpp" "${_stage}/record.json"
    VERBATIM)
  add_dependencies(${target} ${target}_build_inputs)
  target_include_directories(${target} PRIVATE "${_stage}")
  # POST_BUILD refuses drift as part of building the named executable target.
  add_custom_command(TARGET ${target} POST_BUILD
    COMMAND "${Python3_EXECUTABLE}" "${_owner}" publish --root "${root}" --inputs "${_stage}/inputs.json" --stage "${_stage}" --executable "$<TARGET_FILE:${target}>"
    VERBATIM)
  add_custom_target(${target}_build_receipt ALL
    COMMAND "${Python3_EXECUTABLE}" "${_owner}" publish --root "${root}" --inputs "${_stage}/inputs.json" --stage "${_stage}" --executable "$<TARGET_FILE:${target}>"
    DEPENDS ${target}
    VERBATIM)
endfunction()
