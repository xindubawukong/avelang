#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wambiguous-reversed-operator"
#include <mlir/IR/Value.h>
#pragma clang diagnostic pop

#include <optional>

namespace mlir::func {
class FuncOp;
}

namespace causalflow::avelang::ir {

struct TypeInfo {
    std::optional<bool> is_unsigned_integer;
};

// Keep the MLIR representation and its integer interpretation together when
// resolving a declared type. Expression generation still uses mlir::Value.
struct ResolvedType {
    mlir::Type type;
    TypeInfo info;
};

bool IsIntegerValueOrVectorOfIntegers(mlir::Type type);
bool IsMemRefWithIntegerElements(mlir::Type type);

TypeInfo GetTypeInfo(mlir::Value value);
void SetTypeInfo(mlir::Value value, const TypeInfo &typeInfo);
TypeInfo GetFunctionArgumentTypeInfo(mlir::func::FuncOp function,
                                     unsigned index);
TypeInfo GetFunctionResultTypeInfo(mlir::func::FuncOp function, unsigned index);
void SetFunctionResultTypeInfo(mlir::func::FuncOp function, unsigned index,
                               const TypeInfo &typeInfo);

} // namespace causalflow::avelang::ir
