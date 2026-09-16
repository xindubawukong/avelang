add_library(ave-lang-basic
  diagnostic.cc
)

ave_lang_llvm_libraries(llvm_libs support)
target_link_libraries(ave-lang-basic 
  ${llvm_libs} 
  clangBasic
)