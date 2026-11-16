This package contains third-party software components governed by the license(s) indicated below:

---------

Component Name: faer (version 0.19.4)

Source: https://github.com/sarah-quinones/faer-rs

Used in: Unity.Physics/Numerics/Kernels (assembly Unity.Numerics.Kernels): CholeskyLLT.cs, CholeskyLLTFloat.cs,
CholeskyLLTUpdate.cs, Trsm.cs, TrsmFloat.cs, Trsv.cs, TrsvFloat.cs, Trmm.cs, TrmmFloat.cs, Gemv.cs, GemvFloat.cs,
Gemm.cs, GemmFloat.cs, MatrixView.cs, and the dot products and column updates in Dot.cs, DotFloat.cs, Axpy.cs and AxpyFloat.cs. These files
are C# ports of faer's Cholesky (LLT) factor, its solves, its rank update, its insertion and deletion of rows and
columns, the triangular solves that they use, the triangular products, and the shape of faer's matrix views (a pointer, the sizes, and a stride
for rows and one for columns).

License Type: MIT

MIT License

Copyright (c) 2022 sarah

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---------

Component Name: gemm (version 0.18.0)

Source: https://github.com/sarah-ek/gemm

Used in: Unity.Physics/Numerics/Kernels/Gemm.cs and GemmFloat.cs, which follow the shape of gemm's FMA micro-kernels for
doubles (gemm-f64) and floats (gemm-f32).

License Type: MIT

MIT License

Copyright (c) 2021 sarah

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
