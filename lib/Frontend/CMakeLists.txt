ave_lang_llvm_libraries(ave_llvm_libs support)

add_library(ave-lang-frontend
  avelang_parser.cc
  avelang_parser_expr.cc
  avelang_parser_utils.cc
)

target_link_libraries(ave-lang-frontend 
  ave-lang-ast
  ave-lang-basic
  pybind11::pybind11
  Python3::Python
)

add_executable(ave-lang-parser-test avelang_parser_test.cc)
target_link_libraries(ave-lang-parser-test ave-lang-frontend clangFrontend ${ave_llvm_libs} GTest::gtest_main)
add_test(NAME ave-lang-parser-test COMMAND ave-lang-parser-test)
