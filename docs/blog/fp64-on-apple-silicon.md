# How I Got FP64-Class Precision on Apple Silicon GPUs Without FP64 Hardware

The problem seemed impossible when I started: Apple Silicon GPUs can do floating-point math blazingly fast, but they've never supported native 64-bit double precision (FP64). For scientific computing, that's a deal-breaker. Quantum simulations, climate modeling, and other high-precision workloads need the ~15 decimal digits of accuracy that FP64 provides. The GPU's single-precision (FP32) numbers just don't cut it.

Over the past two years, I've been building **apple-bottom**, a BLAS library that delivers FP64-class precision on Apple Silicon Metal GPUs—without any native FP64 hardware. Here's how I did it, why the obvious approaches failed, and what the numbers look like now.

## The Apple Silicon Problem

Let me set the stage. Apple Silicon (M1, M2, M3 families and beyond) is genuinely impressive hardware: fast cores, efficient power, excellent FP32 performance. But Apple omitted FP64 support, likely because the target market (consumer and prosumer devices) has never needed it.

For academic and scientific computing, that's meant:
- No native double-precision matrix operations
- No drop-in compatibility with software like Quantum ESPRESSO, GROMACS, or NumPy's higher-precision modes
- Either living with FP32's 7-digit accuracy, or not using the GPU at all

I wanted to change that.

## Why Obvious Approaches Fail

Before settling on the solution, I tested two alternatives that seemed reasonable but ultimately didn't work.

**Approach 1: GPU Emulation with Accelerate AMX**

The first instinct is to use Apple's own libraries. Accelerate's AMX (Apple Math eXtension) headers promise CPU-based matrix operations at double precision. Benchmarking DGEMM (double-precision general matrix multiply), I got about **536 GFLOP/s** on my M2 Max—not terrible, but it's CPU-only, leaving the GPU idle and missing out on the real performance advantage of moving to GPU compute.

**Approach 2: Integer Emulation (metal-float64)**

Before I started, Philip Turner created [metal-float64](https://github.com/philipturner/metal-float64), which emulates FP64 entirely using integer arithmetic and bit-manipulation in Metal shaders. The idea is clever: simulate floating-point operations with nothing but integer ops and memory access.

The downside? It's slow. I benchmarked it at roughly **24 GFLOP/s**. That's a 22× penalty compared to native FP32 performance, which defeats the purpose of using a GPU.

The fundamental problem: emulating all the components of floating-point arithmetic (normalization, rounding, exponent handling) with integer instructions burns through compute cycles. You're not leveraging the GPU's actual floating-point hardware at all.

## Enter Double-Float (DD) Arithmetic

The breakthrough came when I revisited an old technique from numerical computing: **double-float (DD) arithmetic**.

The idea is beautifully simple: represent each FP64 number as a pair of FP32 values, (hi, lo), such that:
- `hi` holds the most significant bits (the "head")
- `lo` holds the remaining bits (the "tail")
- Together, they give you approximately 48 bits of mantissa

In IEEE 754 terms, a single FP32 has 24 bits of mantissa. By storing two FP32 values and using careful arithmetic, you can get the range of FP64 (exponent ~11 bits) with much higher precision (mantissa ~48 bits) than FP32 alone provides. The relative error target: **~10⁻¹⁵**, matching or exceeding standard FP64.

The key insight: you're not emulating floating-point. You're just using **two FP32 numbers instead of one**, and applying well-established DD arithmetic algorithms. All the hard floating-point work is still done by the GPU's native FP32 hardware.

This sounds simple, but the devil is in the compiler.

## The MTLMathModeSafe Breakthrough

When I started implementing DD arithmetic in Metal shaders, I hit a wall: my precision was catastrophically bad. Intermediate results were getting mangled. The relative errors were ~10⁻⁸ instead of ~10⁻¹⁵.

After weeks of debugging, I realized the problem wasn't my algorithm—it was the **Metal compiler**.

Modern GPU compilers aggressively optimize floating-point math. One common optimization is **Fused Multiply-Add (FMA) reordering**: rearranging operations to maximize FMA instruction usage. In normal FP32 code, this is great for performance with minimal precision loss.

But DD arithmetic is different. The precision comes from careful sequencing of operations. If the compiler reorders your additions or multiplications "for efficiency," it breaks the guarantees you're relying on. An operation like `a + b - c` must happen in exactly that order; reordering to `a - c + b` (mathematically equivalent for unlimited precision) produces different results at finite precision.

The solution: **MTLMathModeSafe**.

Metal's compiler has a flag (`MTLMathMode.safe` in the shader library options) that disables aggressive floating-point optimizations and prevents instruction reordering. It's slightly slower than the default aggressive mode, but for DD arithmetic, it's non-negotiable.

Once I enabled `MTLMathModeSafe`, precision jumped from ~10⁻⁸ to ~10⁻¹⁵. That single compiler flag was the difference between a failed experiment and a working library.

## Performance: 643 GFLOP/s on M2 Max

With DD arithmetic and `MTLMathModeSafe` enabled, I benchmarked DGEMM (multiplying two 8192×8192 matrices) on an M2 Max:

**643 GFLOP/s**

Let's put that in context:
- Accelerate AMX (CPU, FP64): 536 GFLOP/s
- apple-bottom (GPU, DD/FP64): 643 GFLOP/s
- Integer emulation (metal-float64): 24 GFLOP/s

I'm faster than the CPU, and ~25× faster than the previous GPU approach. For scientific code running large matrix operations, that's a real speedup.

Is 643 GFLOP/s close to Apple Silicon's peak FP32 performance? No. With two FP32 values per "FP64," you'd expect roughly half the throughput of pure FP32, and that's roughly what we see. But the point wasn't to match peak FP32—it was to get *usable* double-precision math on a GPU that supposedly doesn't support it.

## Production Validation: Quantum ESPRESSO on Apple Silicon

Numbers on a benchmark slide mean nothing if the code doesn't solve real problems. I needed validation.

I integrated apple-bottom into **Quantum ESPRESSO**, an open-source DFT (Density Functional Theory) code widely used in materials science and quantum chemistry. QE is the gold standard for validating scientific computing libraries: it's production code with a decades-long track record and well-understood expected outputs.

The results were striking:

1. **Performance**: QE running on apple-bottom was **22% faster** than the CPU baseline (Accelerate AMX).

2. **Accuracy**: The computed energies and band structures matched the CPU results to **11 decimal places**. That level of agreement tells you the DD arithmetic is working exactly as the theory predicts.

3. **Drop-in Integration**: Through a Fortran bridge layer, QE saw apple-bottom's DGEMM as just another BLAS implementation. No code changes needed in QE itself. It's a true drop-in replacement.

This wasn't a toy benchmark. This was a 3000+-line scientific code solving real quantum problems. The fact that it ran correctly and faster on the GPU, without modification, proved the approach works.

## What About Triple-Double?

I also explored **triple-double (TD) arithmetic**: using three FP32 values per FP64 number for even higher precision. With TD, I achieved:

- **99.5% correctly-rounded FP64**: For any operation, the result rounds to the nearest representable FP64 value 99.5% of the time. That's nearly perfect rounding.
- **148 GFLOP/s**: Slower than DD (because you're moving three floats instead of two), but still faster than CPU, and exceptional for the precision level.

TD is more of a research direction right now—most users don't need it—but it demonstrates the flexibility of the approach.

## Architecture and Validation

apple-bottom is built to production standards:

- **Metal Shaders**: Optimized kernels for DGEMM, ZGEMM (complex double), and supporting BLAS routines.
- **Fortran Bridge**: A wrapper layer for seamless integration with legacy Fortran code like Quantum ESPRESSO.
- **NASA-STD-7009A Validation**: The library includes verification and validation documentation following NASA's standards for scientific software. This isn't just "it works on my machine"—it's formal, auditable validation.
- **48/48 Tests Pass**: A comprehensive test suite covering precision and correctness, all green.

## The Limitations (Honest Talk)

Let me be clear about what apple-bottom isn't:

- **Not for gaming or graphics**: This is for scientific computing. If you need extreme performance for general FP32 work, native double precision isn't your bottleneck anyway.
- **Not a complete BLAS library yet**: DGEMM and ZGEMM are solid. Other operations are in progress. I'm focused on the workloads that matter most.
- **Not for hardware without a powerful GPU**: apple-bottom shines on M2 Max and higher. Smaller chips with fewer GPU cores will see smaller speedups.
- **MTLMathModeSafe has a cost**: The safe math mode disables compiler optimizations, which means performance is lower than it could theoretically be. For DD arithmetic, correctness wins, but it's a tradeoff.

## Why This Matters

The deeper story here is about **unlocking hardware potential**. Apple Silicon GPUs are genuinely powerful. For years, that power went unused for double-precision scientific workloads because "no FP64 support." But hardware limitations are often more flexible than they first appear.

DD arithmetic is a 30-year-old technique. What made it work *now* was understanding a specific compiler behavior (FMA reordering) and having the right compile flag (`MTLMathModeSafe`) to disable it. The problem and solution were both there; they just needed to be connected.

This has implications beyond apple-bottom. It suggests other "unsupported" hardware features might be unlocked with similar insights. And it underscores why open-source scientific computing matters: I could test against Quantum ESPRESSO, learn from decades of accumulated knowledge about what precision actually means in practice, and build something genuinely useful.

## What's Next

The immediate roadmap:

1. **Broader BLAS Coverage**: Filling in more routines (DGEMV, DGETRF, eigensolvers, etc.) as needed by users.
2. **Optimization**: There's still room to squeeze more performance out of the Metal shaders while maintaining precision. Modern compilers keep improving; we can revisit the optimization strategy.
3. **Broader Hardware Support**: Testing on M3, M4, and other chips as they arrive.
4. **Community**: If you're doing scientific computing on Apple Silicon, try it. File issues. Let me know what doesn't work.

## How to Try It

apple-bottom is open-source and on GitHub. If you're running scientific code on Apple Silicon and hitting the FP64 bottleneck, or if you're curious about how DD arithmetic works, start here:

**[github.com/GrantHeileman/apple-bottom](https://github.com/GrantHeileman/apple-bottom)**

The repo includes:
- Precompiled binaries and source
- The Quantum ESPRESSO integration guide
- Benchmarks you can run on your own hardware
- MIT licensing

Clone it, run `make test`, and see all 48 tests pass on your M-series machine.

## Final Thought

When I started this project, people said "you can't do FP64 on Apple Silicon—there's no hardware." They were technically right. But they missed the distinction between "no hardware" and "no straightforward hardware path." With the right arithmetic technique, the right compiler settings, and careful validation, you can.

That's the real lesson: sometimes the constraints aren't about capability. They're about creativity.

---

*Grant Heileman | April 2026*

*If you found this useful, consider starring the repo or sharing with colleagues doing scientific computing on Mac hardware.*
