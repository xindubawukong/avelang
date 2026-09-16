#pragma once

#include "IR/symbol_scope.h"

namespace causalflow::avelang::ir::intrinsics {

SymbolScope::Function MxFp4ScaledMfma();
SymbolScope::Function MxFp4Pack();
SymbolScope::Function BufferStoreU8();
SymbolScope::Function DsSwizzle();
SymbolScope::Function MaximumF32();

} // namespace causalflow::avelang::ir::intrinsics
