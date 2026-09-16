ave_lang_llvm_libraries(ave_llvm_libs support)

add_library(ave-lang-utils
  embedded_filesystem_view.cc
)

target_link_libraries(ave-lang-utils
  ${ave_llvm_libs}
)