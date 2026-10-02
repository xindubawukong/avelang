#pragma once

#include "type_system.h"

#include <llvm/Support/Error.h>
#include <llvm/Support/JSON.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

namespace causalflow::avelang::ir {

struct ConstexprValue {
    mlir::Attribute value;
    TypeInfo type_info;

    explicit operator bool() const { return bool(value); }
};

// Compile-time data uses builtin attributes, not runtime record types.
llvm::Expected<mlir::Attribute> ParseConstexpr(const llvm::json::Object &info,
                                               mlir::Builder &builder);
mlir::Value MaterializeConstexpr(ConstexprValue value,
                                 mlir::OpBuilder &builder, mlir::Location loc);

} // namespace causalflow::avelang::ir
