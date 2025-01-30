// RUN: circt-opt %s -split-input-file -memory-banking="factors=2,3 dimensions=1" -verify-diagnostics
// RUN: circt-opt %s -split-input-file -memory-banking="factors=2 dimensions=0,1" -verify-diagnostics

// expected-error@+1 {{the number of banking factors must be equal to the number of banking dimensions}}
func.func @factors_gt_dims(%arg0: memref<8x6xf32>) -> (memref<8x6xf32>) {
  %mem = memref.alloc() : memref<8x6xf32>
  affine.parallel (%i) = (0) to (8) {
    affine.parallel (%j) = (0) to (6) {
      %1 = affine.load %arg0[%i, %j] : memref<8x6xf32>
      affine.store %1, %mem[%i, %j] : memref<8x6xf32>
    }
  }
  return %mem : memref<8x6xf32>
}

// -----

// expected-error@+1 {{the number of banking factors must be equal to the number of banking dimensions}}
func.func @dims_gt_factors(%arg0: memref<8x6xf32>) -> (memref<8x6xf32>) {
  %mem = memref.alloc() : memref<8x6xf32>
  affine.parallel (%i) = (0) to (8) {
    affine.parallel (%j) = (0) to (6) {
      %1 = affine.load %arg0[%i, %j] : memref<8x6xf32>
      affine.store %1, %mem[%i, %j] : memref<8x6xf32>
    }
  }
  return %mem : memref<8x6xf32>
}
