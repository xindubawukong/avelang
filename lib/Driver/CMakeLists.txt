ave_lang_llvm_libraries(ave_llvm_libs core support)

add_library(ave-lang-driver
  driver.cc
)

target_link_libraries(ave-lang-driver
  PRIVATE
  ave-lang-frontend
  ave-lang-mlir
  ave-lang-target-gpu
  ave-lang-basic
  ave-lang-ast
  ${ave_llvm_libs}
)

target_compile_definitions(ave-lang-driver PRIVATE ${LLVM_DEFINITIONS})