#pragma once

#include "IR/symbol_scope.h"

namespace causalflow::avelang::ir::intrinsics {

SymbolScope::Function BufferAtomicI32(bool bitwiseOr);
SymbolScope::Function CompilerBarrier();
SymbolScope::Function Sleep();
SymbolScope::Function Fence();

} // namespace causalflow::avelang::ir::intrinsics
