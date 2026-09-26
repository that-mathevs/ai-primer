r"""
# The hardware underneath: chips, memory, links and number formats

Run: `python -m primer.ml.hardware`

## The idea

Every lesson so far has counted operations: so many multiplies per token,
so many parameters. This lesson looks at the machine that performs them,
because the machine explains things the maths alone never will: why a model
that "needs" a tenth of a millisecond of arithmetic takes five milliseconds
per token, why training runs are spread across thousands of chips in a
particular way, and why everyone is shrinking numbers from 32 bits to 8.

Three facts carry the whole lesson:

1. **A GPU is thousands of simple arithmetic units** doing the same step on
   different numbers. Neural networks are mostly matrix multiplies, which
   are exactly that kind of work.
2. **Arithmetic is cheap; moving data is expensive.** The chip can multiply
   far faster than its memory can feed it, so speed is usually decided by
   how many bytes move, not how many operations run.
3. **Fewer bits per number helps everywhere at once**: more numbers per
   byte moved, more numbers in memory, and smaller, more numerous
   multipliers on the chip.

Sections 5 and 6 then apply those facts to many GPUs working together, and
to the question every deployment starts with: will the model fit?

## 1. Why GPUs: thousands of simple cooks

**Everyday picture.** A CPU is a few master chefs. Each can cook anything,
improvise, and follow a recipe full of "if the sauce splits, do this
instead". A GPU is a kitchen of thousands of line cooks who all do the same
step at the same moment on different ingredients: "everyone, chop your
carrot now." That kitchen is useless for inventing a menu and unbeatable at
ten thousand identical salads. A neural network is ten thousand identical
salads: nearly all of its work is **matrix multiplication**, the same
multiply-and-add done billions of times on different numbers.

**Tiny worked example.** Multiply a 2 × 3 matrix by a 3 × 2 matrix:

$$
A = \begin{pmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{pmatrix}, \quad
B = \begin{pmatrix} 7 & 8 \\ 9 & 10 \\ 11 & 12 \end{pmatrix}, \quad
AB = \begin{pmatrix} 58 & 64 \\ 139 & 154 \end{pmatrix}
$$

**Symbols**

| Symbol | Meaning here | Shape |
|---|---|---|
| $A$ | the left matrix: 2 rows of 3 numbers | 2 × 3 |
| $B$ | the right matrix: 3 rows of 2 numbers | 3 × 2 |
| $AB$ | the **matrix multiply**: the cell in row $i$, column $j$ is row $i$ of $A$ dotted with column $j$ of $B$ | 2 × 2 |
| $\begin{pmatrix}\ldots\end{pmatrix}$ | a matrix written out, row by row | |

**In words:** "each cell of the answer is one row of A times one column of
B, multiplied position by position and added up."

**With the numbers:** the top-left cell is 1·7 + 2·9 + 3·11 = 7 + 18 + 33 =
**58**; the bottom-right is 4·8 + 5·10 + 6·12 = 32 + 50 + 72 = **154**. Each
cell took 3 multiplies and 3 additions (each product added to a running
total that starts at 0), so the 4 cells took 12 multiplies and 12 additions.
The crucial detail: **no cell needs any other cell's answer.** Four cooks
could take one cell each and finish at the same moment.

**In Python:**

```python
A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 8], [9, 10], [11, 12]]
m, k, n = len(A), len(B), len(B[0])
# each cell: row i of A dotted with column j of B
[[sum(A[i][p] * B[p][j] for p in range(k)) for j in range(n)] for i in range(m)]  # → [[58, 64], [139, 154]]
```

That count generalises into the most useful formula in this lesson. A
**FLOP** (floating-point operation) is one multiply or one add, and
multiplying an m × k matrix by a k × n matrix costs:

$$
\text{FLOPs} = 2\,m\,n\,k
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $m$ | rows of the left matrix (and of the answer) | 2 |
| $k$ | the shared inner size: columns of the left, rows of the right; the length of each dot product | 3 |
| $n$ | columns of the right matrix (and of the answer) | 2 |
| $m\,n$ | how many cells the answer has | 4 |
| $2$ | one multiply plus one add per step of a dot product | |
| FLOPs | floating-point operations in total | 24 |

**In words:** "every one of the m·n answer cells is a dot product of length
k, and every step of a dot product is a multiply and an add."

**With the numbers:** 2 × 2 × 2 × 3 = **24** for the example. A 4096 × 4096
by 4096 × 4096 multiply, the size of one weight matrix in a mid-sized model,
costs 2 × 4096³ ≈ 1.37 × 10¹¹ FLOPs. At the 10¹⁵ FLOPs per second of the
imaginary datacenter GPU used throughout `primer.ml.inference`, that is
about 0.14 milliseconds, if the chip could be kept busy.

**In Python:**

```python
m, n, k = 2, 2, 3
# 2 FLOPs per step, k steps per cell, m·n cells
2 * m * n * k  # → 24
flops = 2 * 4096 * 4096 * 4096
flops  # → 137438953472
# milliseconds at 10¹⁵ FLOPs per second
round(flops / 1e15 * 1000, 2)  # → 0.14
```

Hardware usually does "multiply, then add to a running total" as a single
instruction, the **fused multiply-add**, which is why FLOPs come in pairs.

```mermaid
flowchart LR
  subgraph CPU["CPU: a few master chefs"]
    direction TB
    c1["core: big control unit,<br/>big cache, runs any code"]
    c2["core"]
    c3["core"]
  end
  subgraph GPU["GPU: thousands of line cooks"]
    direction TB
    g1["group of cores:<br/>one instruction, many numbers"]
    g2["group of cores"]
    g3["... about a hundred groups"]
    g4["matrix units: a small<br/>tile multiply per instruction"]
  end
  W["matrix multiply:<br/>m × n independent cells"] --> GPU
  BR["branchy code:<br/>if this then that"] --> CPU
```

**Reading it:** the two boxes spend their silicon differently. The CPU
spends it on a few cores that each handle any instruction stream quickly,
including code full of decisions. The GPU spends it on arithmetic: groups
of cores that all execute the *same* instruction on *different* numbers,
plus (on most modern accelerators) **matrix units** that multiply a small
tile of numbers in one instruction. A matrix multiply is a perfect fit for
the GPU, because its m × n cells are independent and identical in shape.

How far does the independence go? Suppose each core takes whole cells and
performs one multiply-add per round:

$$
\text{rounds} = \left\lceil \frac{m\,n}{\text{cores}} \right\rceil \times k
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $m\,n$ | independent cells to compute | 64 × 64 = 4,096 |
| cores | identical cores working at once | 1 to 8,192 |
| $\lceil x \rceil$ | **ceiling**: round $x$ up to a whole number (half a cell of work still needs a round) | $\lceil 0.5 \rceil = 1$ |
| $k$ | multiply-adds per cell, done one after another | 64 |
| rounds | how long the whole multiply takes, in rounds | |

**In words:** "share the cells out evenly, round up, and each core then
spends k rounds on each cell it was given."

**With the numbers:** a 64 × 64 by 64 × 64 multiply on 1 core takes
4,096 × 64 = **262,144** rounds; on 64 cores, 4,096; on 4,096 cores, **64**.
On 8,192 cores it is *still* 64: there are only 4,096 cells, so half the
cores have nothing to do.

**In Python:**

```python
import math
m = n = k = 64
def rounds(cores):
    # ⌈m·n / cores⌉ cells per core, then k multiply-adds per cell
    return math.ceil(m * n / cores) * k
rounds(1), rounds(64), rounds(4096), rounds(8192)  # → (262144, 4096, 64, 64)
```

![A 64 by 64 multiply speeds up in a straight line from 1 core to 4,096 cores, 262,144 rounds down to 64, then stays flat because there is no more independent work](figures/primer.ml.hardware.parallel_rounds.svg)

**Reading it:** both axes are logarithmic. Doubling the cores halves the
time, a straight line, until the cores match the number of independent
cells (the dashed line at 4,096). Past that point the line goes flat: more
cores cannot help a problem that has run out of independent work. A small
matrix leaves a big GPU mostly idle.

**In code:** `counted_matmul` multiplies with plain loops and counts every multiply and add; `matmul_flops` is the 2·m·n·k formula; `parallel_rounds` is the rounds formula above.

**Why it matters in practice.** A GPU is fast only when it is given a lot of
independent work at once: large matrices, and many sequences processed
together. That is why serving systems batch requests together
(`primer.ml.inference`) and why a small model answering one user at a time
uses a sliver of the chip. It is also why neural networks look the way they
do: architectures that turn into a few big matrix multiplies (the
transformer, `primer.ml.transformer`) won partly because they suit this
hardware, while step-by-step recurrences (`primer.ml.cnn_rnn`) do not.

## 2. The memory hierarchy: near is small, far is big

**Everyday picture.** Back in the kitchen. A cook's hands hold one or two
things (the **registers**). The cutting board holds a few more (the
**on-chip SRAM**, fast memory built into the chip itself). The fridge in
the kitchen holds the day's ingredients (**HBM**, "high-bandwidth memory",
the GPU's main memory). The storeroom down the hall is the CPU's memory
(**host memory**). The warehouse across town is the **disk**. And other
restaurants' pantries, reached by courier, are other machines over the
**network**. Every step further out holds more and takes longer to reach.
The cooks are fast; what slows the kitchen down is fetching.

**Tiny worked example.** Round, illustrative numbers for one datacenter GPU
and the machine around it (orders of magnitude, not any product's
specification):

| Level | Holds about | Moves about | Streaming 1 GB takes | What lives there |
|---|---|---|---|---|
| registers | 20 MB (across the chip) | 100 TB/s | 0.01 ms | the numbers being multiplied this instant |
| on-chip SRAM | 50 MB | 20 TB/s | 0.05 ms | tiles of the current multiply (section 3) |
| HBM | 80 GB | 3.35 TB/s | 0.30 ms | weights, activations, the KV cache |
| host memory | 1 TB | 50 GB/s (over the link to the GPU) | 20 ms | data waiting to be loaded, offloaded state |
| local disk | 10 TB | 10 GB/s | 100 ms | datasets, checkpoints |
| network | the whole cluster | 50 GB/s per GPU | 20 ms | other GPUs' gradients, remote storage |

Registers and SRAM never hold a whole gigabyte; the column shows their
*rate*. Notice the jump from HBM to host memory: about **67 times** slower.
A GPU that has to reach past its own HBM is a cook walking to the storeroom
for every carrot.

$$
t = \frac{\text{bytes}}{\text{bandwidth}}
$$

**Symbols**

| Symbol | Meaning here | Units |
|---|---|---|
| $t$ | time to stream the data, ignoring the fixed delay before the first byte arrives (latency) | seconds |
| bytes | how much data moves | bytes (1 GB = 10⁹) |
| bandwidth | how many bytes per second the level can deliver | bytes per second |

**In words:** "the time to move data is its size divided by the speed of
the pipe it moves through."

**With the numbers:** an 8-billion-parameter model at 2 bytes per
parameter is 16 GB. Reading it once from HBM takes 16 × 10⁹ / 3.35 × 10¹² =
**4.78 ms**: exactly the lower bound on time per generated token in
`primer.ml.inference`, because generating one token reads every weight once.
From host memory it would take **320 ms**.

**In Python:**

```python
HBM, host = 3.35e12, 50e9
weights = 16e9
# t = bytes / bandwidth, in milliseconds
round(weights / HBM * 1000, 2)  # → 4.78
round(weights / host * 1000)  # → 320
# how many times slower the storeroom is than the fridge
round(HBM / host)  # → 67
```

```mermaid
flowchart LR
  ALU["arithmetic units"] <--> R["registers<br/>~20 MB, ~100 TB/s"]
  R <--> S["on-chip SRAM<br/>~50 MB, ~20 TB/s"]
  S <--> H["HBM<br/>~80 GB, ~3.35 TB/s"]
  H <--> D["host memory<br/>~1 TB, ~50 GB/s"]
  D <--> K["local disk<br/>~10 TB, ~10 GB/s"]
  H <--> N["network: other machines<br/>~50 GB/s per GPU"]
```

**Reading it:** start at the arithmetic units on the left and walk outward.
Every box to the right is bigger and slower. The arithmetic units can only
work on numbers in registers, so every number used must travel the whole
way in from wherever it lives. The network hangs off HBM because fast
clusters let a GPU send and receive data straight from its own memory
without a detour through the CPU.

![Capacity grows about fifty-million-fold from registers to the network while bandwidth falls ten-thousand-fold from registers to disk](figures/primer.ml.hardware.hierarchy.svg)

**Reading it:** the same six levels, top to bottom, on logarithmic axes.
On the left, capacity climbs from megabytes to a petabyte. On the right,
bandwidth falls from a hundred terabytes per second to ten gigabytes per
second. The one place the order bends is the last two rows: a datacenter
network is built to rival a local disk, so fetching from a nearby machine
can be as fast as reading your own drive. Both are still around a hundred
times slower than HBM.

**In code:** `MEMORY_HIERARCHY` lists the six levels with their capacity and bandwidth; `transfer_seconds` is the formula above.

**Why it matters in practice.** A model runs at full speed only if
everything it touches every step (weights, activations, KV cache) lives in
HBM. Spilling to host memory ("offloading") makes a model fit, at the price
of each step waiting tens of times longer. And because HBM itself is slow
compared with the arithmetic, the fastest code is the code that makes each
trip to HBM count, which is the next section.

## 3. Arithmetic intensity: why data movement dominates

**Everyday picture.** A sandwich shop. If the cook walks to the storeroom
for each slice of bread for each sandwich, the cook spends the day walking.
If the cook carries a tray of bread and a tray of fillings to the bench and
makes a batch of sandwiches from them, every trip feeds many sandwiches.
Same sandwiches (FLOPs), far fewer trips (bytes). The ratio of the two is
the **arithmetic intensity**: operations done per byte fetched.

Our imaginary GPU does 10¹⁵ FLOPs per second but reads only 3.35 × 10¹²
bytes per second from HBM, so it breaks even at about **299 FLOPs per
byte** (the ridge point of the roofline in `primer.ml.inference`). Any work
doing fewer operations than that for each byte it fetches leaves the
arithmetic units waiting on memory.

**Tiny worked example.** Multiply two 4 × 4 matrices: 2 × 4³ = 128 FLOPs.
Count the numbers fetched from slow memory (HBM) into fast memory (SRAM):

| Strategy | Numbers read | Numbers written | FLOPs per number moved |
|---|---|---|---|
| no reuse: each cell fetches its own row of A and column of B | 16 × (4 + 4) = **128** | 16 | 128 / 144 = 0.89 |
| 2 × 2 tiles: load a tile of A and a tile of B, use each number twice | **64** | 16 | 128 / 80 = 1.6 |
| one 4 × 4 tile: load everything once | **32** | 16 | 128 / 48 = 2.7 |

The arithmetic is identical in all three rows. Only the traffic changes.
This trick is **tiling**: bring a small block of each matrix into fast
memory and do every multiplication that block takes part in before
throwing it away.

$$
\text{reads} = \frac{2\,n^3}{T}
\qquad
I = \frac{2\,n^3}{b\left(\dfrac{2\,n^3}{T} + n^2\right)} \approx \frac{T}{b}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $n$ | both matrices are $n \times n$ | 4, then 4,096 |
| $T$ | tile width: fast memory works on $T \times T$ blocks | 1, 2, 4, then 128 |
| $2\,n^3$ | the FLOPs of the whole multiply (section 1, with $m = n = k$) | 128 |
| reads | numbers fetched from slow memory | 128, 64, 32 |
| $n^2$ | numbers written back: each answer cell once | 16 |
| $b$ | bytes per number: 2 at 16-bit, 1 at 8-bit | 2 |
| $I$ | arithmetic intensity: FLOPs per byte moved | FLOPs/byte |
| $\approx$ | "roughly", once $n$ is much bigger than $T$ and the writes are negligible | |

**In words:** "every number fetched is used T times, so the traffic falls in
proportion to the tile width, and the intensity rises in proportion to it."

**With the numbers:** for n = 4, the reads are 2 × 64 / T = 128, 64 and 32
for tiles of 1, 2 and 4, as the table says. For two 4096 × 4096 matrices in
16-bit with 128-wide tiles, I ≈ 128 / 2 = 64, and **63.0** once the
writes are counted. In 8-bit the same tiles give **126**: halving the bytes
per number doubles the intensity.

**In Python:**

```python
n = 4
# reads = 2n³ / T, for tiles of 1, 2 and 4
[2 * n**3 // T for T in (1, 2, 4)]  # → [128, 64, 32]
def intensity(n, T, b):
    flops = 2 * n**3
    # bytes moved: every read, plus one write per answer cell
    moved = b * (2 * n**3 / T + n**2)
    return flops / moved
round(intensity(4096, 128, 2), 1)  # → 63.0
round(intensity(4096, 128, 1), 1)  # → 126.0
```

```mermaid
flowchart LR
  subgraph HBM["HBM: big, slow"]
    A["A, in T × T tiles"]
    B["B, in T × T tiles"]
    C["C, the answer"]
  end
  subgraph SRAM["on-chip SRAM: small, fast"]
    a["one tile of A"]
    b["one tile of B"]
    acc["running total for<br/>one T × T tile of C"]
  end
  A -- "load" --> a
  B -- "load" --> b
  a --> mm["multiply-add:<br/>T³ steps, no traffic"]
  b --> mm
  mm --> acc
  mm -. "next pair of tiles along k" .-> A
  acc -- "write once, at the end" --> C
```

**Reading it:** the left box is slow memory, the right box fast memory.
For one tile of the answer, the kernel repeatedly loads one tile of A and
one tile of B, does all T³ multiply-adds between them without touching slow
memory, and adds the results into a running total that stays on chip. Only
when the whole row of tiles has been consumed does it write the finished
answer tile back, once. Fast memory never holds more than three tiles.

![Measured reads fall from 65,536 to 2,048 as the tile grows from 1 to 32, on the 2n-cubed-over-T line, and intensity at n = 4096 rises with the tile, reaching the 299 break-even only near T = 650 in 16-bit or T = 310 in 8-bit](figures/primer.ml.hardware.tiling.svg)

**Reading it:** on the left, dots are reads counted by actually running the
tiled multiply on 32 × 32 matrices, and the line is the formula 2n³/T; they
agree exactly, and each doubling of the tile halves the traffic. On the
right, the intensity of a 4096 × 4096 multiply climbs with the tile width,
and the dashed line is the chip's break-even point of 299. In 16-bit the
tiles need to be about 650 wide to cross it. Three 650 × 650 tiles in 16-bit
take about 2.5 MB, more fast memory than one group of cores has to itself,
which is why real kernels tile at several levels at once (tiles in
registers inside tiles in SRAM) and why lower-precision numbers, which
shift the whole curve up, are so attractive.

The same idea runs through the rest of this primer:

- **FlashAttention** (`primer.ml.attention`) tiles attention: blocks of
  queries, keys and values are loaded into SRAM, and the n × n score matrix
  is never written to HBM at all. Same answer, a fraction of the traffic.
- **Kernel fusion**: adding a bias or applying an activation does about one
  FLOP per number it reads, hopelessly below 299, so these steps are done
  inside the matrix-multiply kernel while the tile is still on chip.
- **Decode** (`primer.ml.inference`): generating one token for one user
  reads every weight to do just 2 FLOPs with it, an intensity of about 1.
  Batching users together is tiling across requests: one read of a weight
  serves every sequence in the batch.

**In code:** `tiled_matmul` runs the tiled multiply and returns a `Traffic` count of reads, writes, FLOPs and peak fast-memory use; `matmul_reads` and `matmul_intensity` are the formulas; `primer.ml.inference.ridge_point` is the break-even.

**Why it matters in practice.** Before asking how many FLOPs a piece of
work needs, ask how many bytes it moves and how often each byte is reused.
Most large speed-ups in modern AI systems (FlashAttention, fused kernels,
batching, quantization) change the bytes, not the FLOPs.

## 4. Number formats: how many bits each number gets

**Everyday picture.** Scientific notation on a form with a fixed number of
boxes: 6.02 × 10²³. One box holds the sign. A few boxes hold the power of
ten, which sets how big or small the number can be: its **range**. The
rest hold the digits, which set how finely it is measured: its
**precision**. With a fixed number of boxes, moving a box from the digits
to the power buys range and costs precision. Computers do the same with
bits and powers of two, and call the parts the **sign**, the **exponent**
and the **mantissa** (the stored digits).

**Tiny worked example.** Store −6.5 in **bf16** ("brain float 16": 1 sign
bit, 8 exponent bits, 7 mantissa bits).

1. **Sign:** negative, so the sign bit is 1.
2. **Power of two:** the largest power of two not above 6.5 is 4 = 2², so
   6.5 = 1.625 × 2².
3. **Exponent:** stored with a **bias** of 127 added, so that negative
   powers need no sign of their own: 2 + 127 = 129 = 10000001 in binary.
4. **Mantissa:** the leading 1 of 1.625 is always there, so it is not
   stored. The fraction 0.625 = ½ + ⅛ is 0.101 in binary, padded to seven
   bits: 1010000.
5. **The 16 bits:** 1 10000001 1010000.

Most numbers are not so lucky. 0.1 has no finite binary expansion, so it is
rounded to the nearest value each format can hold: 0.10000000149 in fp32,
0.1000977 in bf16, 0.0999756 in fp16 and 0.1015625 in 8-bit E4M3.

$$
x = (-1)^{s} \times 2^{\,e - \text{bias}} \times \left(1 + \frac{f}{2^{M}}\right),
\qquad \text{bias} = 2^{E-1} - 1
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $s$ | the sign bit: 0 positive, 1 negative | 1 |
| $(-1)^s$ | −1 multiplied by itself $s$ times: +1 when $s = 0$, −1 when $s = 1$ | −1 |
| $E$ | how many exponent bits the format has | 8 |
| $e$ | the stored exponent, read as an ordinary whole number | 129 |
| bias | the offset subtracted from $e$, so stored values 1…254 stand for powers −126…127 | 127 |
| $M$ | how many mantissa bits the format has | 7 |
| $f$ | the stored mantissa, read as a whole number from 0 to $2^M - 1$ | 1010000 = 80 |
| $1 + f/2^M$ | the significand: the hidden leading 1 plus the stored fraction, between 1 and 2 | 1 + 80/128 = 1.625 |
| $x$ | the number the bits stand for | −6.5 |

**In words:** "the sign says plus or minus, the exponent says which power of
two to scale by, and the mantissa says how far between that power and the
next one the number sits."

**With the numbers:** (−1)¹ × 2^(129 − 127) × (1 + 80/128) = −1 × 4 × 1.625 =
**−6.5**.

**In Python:**

```python
x = -6.5
M, bias = 7, 127
# s: 1 for a negative number
s = 1 if x < 0 else 0
# e: the power of two below |x| is 2², stored with the bias added
e = 2 + bias
e, format(e, "08b")  # → (129, '10000001')
# f: the fraction after the hidden 1 of |x| / 2², as a 7-bit whole number
f = round((abs(x) / 2**2 - 1) * 2**M)
f, format(f, "07b")  # → (80, '1010000')
# decode: (-1)^s × 2^(e - bias) × (1 + f / 2^M)
(-1)**s * 2**(e - bias) * (1 + f / 2**M)  # → -6.5
```

Two corners of the formula matter in practice. When the stored exponent is
0, the hidden 1 is dropped and the number is a **subnormal**: it lets values
fade gradually towards zero instead of dropping off a cliff, at the cost of
fewer significant bits. And IEEE-style formats reserve the all-ones
exponent for **infinity** and **NaN** ("not a number"), which is where
overflowing values go.

```mermaid
flowchart LR
  X["x = −6.5"] --> S["sign: negative<br/>s = 1"]
  X --> P["largest power of two<br/>not above 6.5: 2² = 4"]
  P --> E["exponent: 2 + bias 127<br/>e = 129 = 10000001"]
  P --> F["6.5 / 4 = 1.625<br/>drop the leading 1: .625"]
  F --> R["round .625 to 7 bits<br/>f = 1010000"]
  S --> B["1 | 10000001 | 1010000"]
  E --> B
  R --> B
```

**Reading it:** a number enters on the left and splits three ways. The sign
is read off directly. The exponent comes from finding which pair of powers
of two the number sits between. The mantissa is where the number sits
within that pair, rounded to however many bits the format allows: this
rounding box is the only place information is lost, and it is where every
format differs.

![Bit layouts drawn to scale: fp32 has 1 sign, 8 exponent and 23 mantissa bits; bf16 keeps the 8 exponent bits and cuts the mantissa to 7; fp16 has 5 and 10; the two fp8 formats have 5 and 2, or 4 and 3; int8 and int4 are plain integers](figures/primer.ml.hardware.format_layouts.svg)

**Reading it:** each row is one format drawn to scale, one cell per bit,
with the sign in grey, the exponent in orange and the mantissa in blue.
Compare bf16 and fp16: the same 16 bits, split differently. bf16 is
literally the top half of fp32, keeping all 8 exponent bits (fp32's whole
range) and giving up precision; fp16 keeps more precision and gives up
range. The integer formats at the bottom have no exponent at all: every
value is a whole number, turned into a weight by one shared scale
(`primer.ml.inference` builds int8 and int4 quantization from scratch).

| Format | Bits (sign, exponent, mantissa) | Largest | Smallest at full precision | Gap just above 1 | Typical use |
|---|---|---|---|---|---|
| fp32 | 1, 8, 23 | 3.4 × 10³⁸ | 1.2 × 10⁻³⁸ | 1.2 × 10⁻⁷ | master weights, optimizer state, running sums |
| bf16 | 1, 8, 7 | 3.4 × 10³⁸ | 1.2 × 10⁻³⁸ | 1/128 ≈ 0.0078 | training and inference matrix multiplies |
| fp16 | 1, 5, 10 | 65,504 | 6.1 × 10⁻⁵ | 1/1024 ≈ 0.00098 | inference; training with loss scaling |
| fp8 E5M2 | 1, 5, 2 | 57,344 | 6.1 × 10⁻⁵ | 0.25 | gradients in 8-bit training |
| fp8 E4M3 | 1, 4, 3 | 448 | 0.0156 | 0.125 | weights and activations in 8-bit |
| int8 | 8-bit integer | 127 × scale | evenly spaced | none: a fixed step | quantized weights |
| int4 | 4-bit integer | 7 × scale | evenly spaced | none: a fixed step | quantized weights |

E4M3 bends the IEEE rules: it has no infinity, and spends that exponent on
ordinary numbers instead, which is how 8 bits reach 448 rather than 240.

![Relative spacing between neighbouring values: each float format is a flat band across its range (fp32 near 1e-7, bf16 near 1e-2, fp8 near 0.1), rising at its small end and stopping at its largest value, while int8 and int4 spacing rises steadily as numbers shrink](figures/primer.ml.hardware.precision.svg)

**Reading it:** the x-axis is the size of the number being stored; the
y-axis is the gap to the next storable value, as a fraction of the number
(lower is more precise); both are logarithmic. Each float format is a flat
band: its relative precision is the same for tiny and huge numbers, which
is the whole point of an exponent. The band's width is the range: it stops
on the right at the largest value (65,504 for fp16, 448 for E4M3) and rises
on the left where subnormals run out of bits. fp32 and bf16 are flat across
the whole plot and far beyond it. The integer formats, here with a scale
that maps 1.0 to the top code, are straight rising lines: their step is the
same size everywhere, so small numbers get coarse, which is why quantized
models use one scale per row or block of weights.

What the picture means for training (`primer.ml.pretraining` covers mixed
precision in full):

- **Range failures.** A gradient of 10⁻⁸ becomes exactly 0 in fp16 (below
  its smallest subnormal) but survives in bf16 as 1.0012 × 10⁻⁸. fp16
  training therefore multiplies the loss by a large constant (**loss
  scaling**) to lift gradients into range; bf16 training does not need to.
- **Precision failures.** In bf16, 1 + 0.001 rounds back to exactly 1: a
  small weight update simply vanishes. So training keeps a **master copy**
  of the weights in fp32, adds its sums up in fp32, and uses 16-bit (or
  8-bit) only for the big multiplies. That split is **mixed precision**.

### Why smaller formats multiply throughput

Fewer bits per number pays three times:

1. **Bytes.** Half the bytes per number moves twice the numbers per second
   through every level of section 2, and fits twice the parameters in HBM.
   Memory-bound work speeds up directly: the lower bound on time per token
   for a 70-billion-parameter model (`primer.ml.inference`) is 41.8 ms at
   16-bit, 20.9 ms at 8-bit and 10.4 ms at 4-bit.
2. **Intensity.** The same tile does twice the FLOPs per byte (section 3:
   63 becomes 126), pushing more work past the break-even point.
3. **Silicon.** A multiplier is the expensive part of the chip, and its
   size grows with the square of the number of significand bits.

**Everyday picture.** Long multiplication by hand: multiplying two 3-digit
numbers means writing a 3 × 3 grid of single-digit products; two 6-digit
numbers need a 6 × 6 grid, four times the work. A chip's multiplier is that
grid built in wires.

$$
\text{cells} = p^{2}, \qquad p = M + 1
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $M$ | stored mantissa bits | 23, 10, 7, 3 |
| $p$ | significand bits: the stored mantissa plus the hidden leading 1 | 24, 11, 8, 4 |
| cells | one-bit products in a schoolbook (array) multiplier: one per pair of bits | |

**In words:** "multiplying two p-bit significands needs one small cell for
every pair of bits, so p times p cells; the exponents only need adding,
which is cheap."

**With the numbers:** fp32 needs 24² = **576** cells, fp16 11² = 121, bf16
8² = 64, and fp8 E4M3 4² = **16**: one fp32 multiplier's worth of silicon
holds many 8-bit ones. Accelerators typically list roughly double the peak
operations per second each time the format halves.

**In Python:**

```python
# stored mantissa bits for fp32, fp16, bf16 and fp8 E4M3
mantissa_bits = [23, 10, 7, 3]
# p = M + 1, and cells = p²
[(M + 1) ** 2 for M in mantissa_bits]  # → [576, 121, 64, 16]
```

**In code:** `FloatFormat` describes a format by its exponent and mantissa bits, with `FloatFormat.max_value`, `FloatFormat.min_normal`, `FloatFormat.min_subnormal` and `FloatFormat.epsilon` derived from them; `FP32`, `BF16`, `FP16`, `FP8_E5M2` and `FP8_E4M3` are the five formats; `encode` rounds a number into its three fields with plain arithmetic, `decode` turns fields back into a number, `round_to` does both, and `bit_string` prints the bits; `multiplier_cells` is the p² formula. The tests check `round_to` against NumPy's own float16 and float32.

**Why it matters in practice.** Picking a number format is picking a point
on the range-versus-precision curve for each kind of number in a model.
Weights and activations tolerate coarse formats; gradients need range; the
running sums of long dot products need precision. Modern training and
serving use a different format for each, and the savings are among the
largest in the field.

## 5. Many GPUs: the cost of talking

**Everyday picture.** A group project. Four people each work through a
quarter of the exercises, and then must agree on one combined answer
sheet. Sitting at the same table they can compare notes in seconds; living
in different cities they must post letters. The more often a team must
compare notes, the more it matters who sits at the same table.

Large models are trained on many GPUs because no single one has the memory
or the speed. The simplest split is **data parallelism**: every GPU holds a
copy of the model and works on different examples, and after each step
their gradients must be added up, so every copy takes the same step. That
"add up, and give everyone the total" operation is an **all-reduce**.

**Tiny worked example: a ring all-reduce.** Four GPUs each hold a gradient
of 8 numbers. Each splits its gradient into 4 chunks of 2 numbers and they
sit in a ring, each passing to its right-hand neighbour.

1. **Reduce-scatter, 3 steps:** each GPU sends one chunk to its neighbour,
   which adds it to its own copy of that chunk. After 3 steps, each GPU
   holds one chunk that contains the sum from all four.
2. **All-gather, 3 more steps:** the finished chunks travel round the ring,
   overwriting the stale copies.

Each GPU sent 6 chunks of 2 numbers, **12 numbers**, which is
2 × 3/4 × 8. The remarkable part: with 400 GPUs instead of 4, each would
still send just under twice its gradient. The work per GPU barely grows.

```mermaid
flowchart LR
  G0["GPU 0<br/>chunks a b c d"] -- "one chunk per step" --> G1["GPU 1<br/>chunks a b c d"]
  G1 -- "one chunk per step" --> G2["GPU 2<br/>chunks a b c d"]
  G2 -- "one chunk per step" --> G3["GPU 3<br/>chunks a b c d"]
  G3 -- "one chunk per step" --> G0
```

**Reading it:** four GPUs in a ring, each only ever talking to its
right-hand neighbour. Every link is busy at every step, each carrying a
different chunk, so no link sits idle and no single GPU becomes a
bottleneck. After 2 × (4 − 1) = 6 steps, every GPU has every chunk summed
over all four.

$$
t_{\text{all-reduce}} = \frac{2\,(N-1)}{N} \cdot \frac{S}{B}
\qquad
t_{\text{compute}} = \frac{6\,P\,D}{F}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $N$ | GPUs taking part | 8 |
| $S$ | bytes of gradient each GPU holds | 7 × 10⁹ parameters × 2 bytes = 14 GB |
| $B$ | each GPU's link bandwidth | 500 GB/s in one machine, 50 GB/s between machines (illustrative) |
| $\frac{2(N-1)}{N}$ | the fraction of its gradient each GPU sends in a ring all-reduce: just under 2 | 1.75 |
| $P$ | parameters in the model | 7 × 10⁹ |
| $D$ | tokens each GPU processes per step | 16,384 |
| $6$ | FLOPs per parameter per token in training: 2 forward, 4 backward | |
| $F$ | the GPU's arithmetic speed | 10¹⁵ FLOPs per second |

**In words:** "talking takes just under twice the gradient's size divided by
the link speed, however many GPUs there are; computing takes six operations
per parameter per token, divided by the chip's speed."

**With the numbers:** inside one machine, 1.75 × 14 × 10⁹ / 500 × 10⁹ =
**49 ms**. Across machines, **490 ms**. The arithmetic for the step is
6 × 7 × 10⁹ × 16,384 / 10¹⁵ = **0.69 s**. Inside a machine, talking costs 7%
of the computing time; across machines, 71%.

**In Python:**

```python
N, S = 8, 14e9
in_machine, between_machines = 500e9, 50e9
# 2(N - 1)/N × S / B, in milliseconds
round(2 * (N - 1) / N * S / in_machine * 1000)  # → 49
round(2 * (N - 1) / N * S / between_machines * 1000)  # → 490
P, D, F = 7e9, 16_384, 1e15
# 6·P·D / F, in seconds
round(6 * P * D / F, 2)  # → 0.69
```

![All-reduce time levels off as GPUs are added: about 56 ms over fast in-machine links and about 560 ms over the network, against 690 ms of arithmetic per step](figures/primer.ml.hardware.communication.svg)

**Reading it:** the x-axis is the number of GPUs (log scale); the y-axis
is seconds per training step. The flat grey line is the arithmetic every
GPU does per step. The two rising curves are the all-reduce over each kind
of link, and both flatten out almost at once: that is the ring's
2(N − 1)/N approaching 2. The gap between the curves is the whole story:
the same gradient costs ten times more over the network than over in-machine
links, bringing communication close to the cost of the arithmetic itself.
Real systems hide part of it by sending the gradients of later layers while
earlier layers are still computing theirs.

```mermaid
flowchart TB
  subgraph M1["machine 1: fast links, ~500 GB/s"]
    a1["GPU"] <--> a2["GPU"] <--> a3["GPU"] <--> a4["GPU"]
  end
  subgraph M2["machine 2: fast links, ~500 GB/s"]
    b1["GPU"] <--> b2["GPU"] <--> b3["GPU"] <--> b4["GPU"]
  end
  M1 <-- "network, ~50 GB/s per GPU:<br/>data parallelism, once per step" --> M2
  TP["tensor parallelism:<br/>talks inside every layer"] -.-> M1
  TP -.-> M2
```

**Reading it:** two machines, each with a few GPUs joined by fast links, and
a slower network between them. The chattiest kind of splitting, **tensor
parallelism** (cutting each matrix multiply across GPUs, which must swap
partial results inside every layer), is kept within one machine's fast
links. Kinds that talk rarely, like data parallelism (one all-reduce per
step) and **pipeline parallelism** (passing activations only between
neighbouring stages of layers), span the slower network. The layout of the
wires decides the layout of the work. `primer.ml.pretraining` builds these
strategies in full.

**In code:** `ring_all_reduce` simulates the ring step by step and counts the numbers each GPU sends; `all_reduce_seconds` and `training_step_seconds` are the two formulas; `IN_MACHINE_LINK` and `BETWEEN_MACHINES_LINK` are the illustrative link speeds.

**Why it matters in practice.** At scale, a cluster's network matters as
much as its chips. Parallelism strategies are chosen by matching how often
each kind talks to how fast each link is, and a training run that ignores
the map spends its budget waiting.

## 6. Will it fit? Memory math for serving

**Everyday picture.** A bookshelf of fixed width. The encyclopedia (the
model's weights) must go on it, whole. Whatever space is left holds one
notebook per customer being served (their KV cache), and a notebook grows
with every word of the conversation. Thinner volumes, printed in a smaller
number format, leave room for more notebooks.

**Tiny worked example.** A model shaped like Llama 3 70B (80 layers, 8
key/value heads of 128 dimensions each) on one 80 GB GPU. Its KV cache
costs 2 × 80 × 8 × 128 × 2 = 327,680 bytes per token at 16-bit
(`primer.ml.inference` derives this).

| Weights | Weight memory | Left for the cache | Tokens of 16-bit cache | Tokens of 8-bit cache |
|---|---|---|---|---|
| 16-bit | 140 GB | none: does not fit | 0 | 0 |
| 8-bit (fp8) | 70 GB | 10 GB | 30,517 | 61,035 |
| 4-bit | 35 GB | 45 GB | 137,329 | 274,658 |

This ignores activations and the serving software's own working memory,
which take several more gigabytes in practice.

$$
P \, b_w + C \times 2\,L\,H_{kv}\,d_h\,b_{kv} \le M_{\text{GPU}}
$$

**Symbols**

| Symbol | Meaning here | In the example |
|---|---|---|
| $P$ | parameters | 7 × 10¹⁰ |
| $b_w$ | bytes per weight: bits ÷ 8 | 2, 1 or 0.5 |
| $C$ | tokens held in the KV cache, summed over every request being served | the unknown |
| $2$ | one key and one value per token | |
| $L$ | layers, each with its own cache | 80 |
| $H_{kv}$ | key/value heads | 8 |
| $d_h$ | dimensions per head | 128 |
| $b_{kv}$ | bytes per cached number | 2 or 1 |
| $M_{\text{GPU}}$ | the GPU's memory | 80 × 10⁹ bytes |
| $\le$ | "must be at most" | |

**In words:** "the weights, plus every cached token's keys and values
across every layer, must fit in the GPU's memory."

**With the numbers:** with fp8 weights, 7 × 10¹⁰ × 1 = 70 GB, leaving 10 GB.
10 × 10⁹ / 327,680 = **30,517** tokens of 16-bit cache, or **61,035** with
the cache in fp8 too: one long conversation, or a dozen short ones.

**In Python:**

```python
P, M_gpu = 70e9, 80e9
L, H_kv, d_h = 80, 8, 128
def kv_per_token(b_kv):
    # a key and a value, per layer, per KV head
    return 2 * L * H_kv * d_h * b_kv
kv_per_token(2)  # → 327680
# weight GB at 16, 8 and 4 bits
[P * bits / 8 / 1e9 for bits in (16, 8, 4)]  # → [140.0, 70.0, 35.0]
# tokens of cache beside fp8 weights, with a 16-bit cache and then an fp8 one
int((M_gpu - P * 1) // kv_per_token(2)), int((M_gpu - P * 1) // kv_per_token(1))  # → (30517, 61035)
```

```mermaid
flowchart LR
  P["parameters × bytes per weight"] --> W["weights"]
  T["tokens × bytes per token"] --> K["KV cache"]
  W --> Q{"weights + cache<br/>fit in GPU memory?"}
  K --> Q
  Q -- yes --> Y["serve: spare room means<br/>more users or longer contexts"]
  Q -- no --> N["smaller formats, fewer KV heads,<br/>shorter contexts, or more GPUs"]
```

**Reading it:** two budgets feed one question. The weights are a fixed cost,
set by the parameter count and the number format. The cache grows with
every token of every active conversation. If they fit together, whatever is
left over becomes capacity: more concurrent users, longer contexts. If
they don't, every lever on the right shrinks one of the two inputs, or
splits the model across GPUs joined by the fast links of section 5.

![Stacked bars against an 80 GB line: 16-bit weights alone reach 140 GB and overflow; fp8 weights take 70 GB and leave 10 GB, about 30,000 tokens of cache; 4-bit weights take 35 GB and leave 45 GB, about 137,000 tokens](figures/primer.ml.hardware.serving_fit.svg)

**Reading it:** each bar is one choice of weight format for the same
70-billion-parameter model. The dark part is the weights; the light part
is the room left for the KV cache, labelled with how many 16-bit tokens fit
in it. The dashed line is the GPU's 80 GB. At 16-bit the weights alone
break through the line. Each halving of the weight format frees memory that
turns directly into tokens of cache, which is to say into users.

**In code:** `max_cache_tokens` subtracts the weights and divides what is left by the per-token cache, using `primer.ml.inference.weight_bytes` and `primer.ml.inference.kv_cache_bytes_per_token`; `LLAMA3_70B_SHAPE` holds the example model's shape.

**Why it matters in practice.** This arithmetic comes first in every
deployment: the number format sets whether a model fits, how many users
share a GPU, and (because decode reads every weight per token) how fast
each of them sees words appear. `primer.ml.inference` carries it on into
batching, speculative decoding and prompt caching.

## In 20 seconds

- **GPUs** are thousands of simple cores doing the same step on different
  numbers. Matrix multiplies (2·m·n·k FLOPs, every cell independent) are
  the perfect workload, as long as there is enough independent work.
- **Memory hierarchy:** registers, on-chip SRAM, HBM, host memory, disk,
  network. Each step out is bigger and slower; the GPU runs at full speed
  only on data in HBM or closer.
- **Arithmetic intensity:** the chip does hundreds of operations in the
  time it fetches one byte, so speed is set by bytes moved. Tiling reuses
  each fetched number T times and cuts traffic T-fold; FlashAttention,
  kernel fusion and batching are the same idea.
- **Number formats:** sign, exponent (range) and mantissa (precision).
  bf16 keeps fp32's range with less precision; fp16 the reverse; fp8 and
  int8/int4 trade more. Fewer bits means fewer bytes, higher intensity and
  smaller multipliers, so throughput roughly doubles per halving.
- **Many GPUs:** a ring all-reduce costs about 2 × gradient size ÷ link
  speed, whatever the GPU count. Chatty parallelism stays on fast
  in-machine links; the rest crosses the network.
- **Serving:** weights plus KV cache must fit; the number format decides
  both what fits and how fast each token comes.

## Self-test questions

**Why are GPUs, rather than CPUs, used to train and run neural networks?**
Nearly all of a network's work is matrix multiplication, where every output
cell is an independent dot product. A GPU spends its silicon on thousands
of simple arithmetic units (plus matrix units) that apply the same
instruction to different numbers, so it can compute thousands of cells at
once. A CPU spends its silicon on a few flexible cores that are better at
branchy, sequential code.

**How many FLOPs does multiplying a 1,000 × 2,000 matrix by a 2,000 × 500 matrix take, and why that formula?**
2 × 1,000 × 500 × 2,000 = 2 × 10⁹. There are m·n = 500,000 output cells,
each a dot product of length k = 2,000, and each step of a dot product is
one multiply and one add.

**The chip can do 10¹⁵ FLOPs per second but a model runs far slower. What is usually the bottleneck, and how do you tell?**
Memory bandwidth. Compare the work's arithmetic intensity (FLOPs per byte
moved from HBM) with the chip's break-even ratio, peak FLOPs divided by
bandwidth (about 299 here). Below it, the arithmetic units wait on memory;
generating one token for one user has an intensity near 1, so it is
deeply memory-bound.

**How does tiling a matrix multiply reduce memory traffic, and what limits it?**
Each block of numbers is loaded into fast on-chip memory once and used for
every multiplication it takes part in, T times, instead of being fetched
again for each one. Reads fall from 2n³ to 2n³/T. The limit is fast-memory
size: three T × T tiles must fit at once, so real kernels tile at several
levels of the hierarchy.

**What is the difference between bf16 and fp16, and why do many training runs prefer bf16?**
Both have 16 bits. bf16 has fp32's 8 exponent bits and 7 mantissa bits:
the same range as fp32, less precision. fp16 has 5 exponent bits and 10
mantissa bits: more precision, but a range that tops out at 65,504 and
rounds gradients smaller than about 3 × 10⁻⁸ to zero. bf16 avoids the need for
loss scaling; its coarse precision is handled by keeping fp32 master
weights and fp32 sums.

**Why does halving the bits per number roughly double throughput?**
Three reasons: half the bytes to move, so memory-bound work runs twice as
fast and twice the parameters fit; twice the FLOPs per byte for the same
tile, so more work clears the break-even point; and multipliers whose area
grows with the square of the significand bits, so many more small
multipliers fit in the same silicon.

**Why is tensor parallelism usually kept inside one machine while data parallelism spans many?**
Tensor parallelism splits every matrix multiply, so GPUs must exchange
partial results inside every layer, many times per step: it needs the fast
in-machine links. Data parallelism communicates once per step (an
all-reduce of the gradients), which a ring spreads so each GPU sends only
about twice its gradient, and which can overlap with the backward pass, so
it tolerates the slower network.

**Will a 70-billion-parameter model serve from one 80 GB GPU?**
Not in 16-bit: the weights alone are 140 GB. In 8-bit the weights take 70
GB, leaving about 10 GB, around 30,000 tokens of 16-bit KV cache for a
model with 80 layers and 8 KV heads of 128 dimensions. In 4-bit, 45 GB is
left, about 137,000 tokens. Leave headroom for activations and the serving
software.

## The papers behind this lesson

- **Williams, Waterman and Patterson, *Roofline: An Insightful Visual
  Performance Model for Multicore Architectures* (2009)**:
  https://doi.org/10.1145/1498765.1498785. Introduced the roofline: judge a
  kernel by its arithmetic intensity against the machine's balance of
  compute and bandwidth.
- **Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention
  with IO-Awareness* (2022)**: https://arxiv.org/abs/2205.14135. Applied
  tiling to attention so the score matrix never reaches HBM, showing that
  counting memory traffic rather than FLOPs is what makes attention fast.
- **Micikevicius et al., *Mixed Precision Training* (2017)**:
  https://arxiv.org/abs/1710.03740. Showed that networks train in 16-bit
  floats with fp32 master weights, fp32 accumulation and loss scaling.
- **Kalamkar et al., *A Study of BFLOAT16 for Deep Learning Training*
  (2019)**: https://arxiv.org/abs/1905.12322. Showed that bf16, with fp32's
  range, trains a wide range of models to fp32 quality without loss scaling.
- **Micikevicius et al., *FP8 Formats for Deep Learning* (2022)**:
  https://arxiv.org/abs/2209.05433. Proposed the E4M3 and E5M2 8-bit
  formats and showed that training and inference hold up in them.
- **Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language
  Models Using Model Parallelism* (2019)**: https://arxiv.org/abs/1909.08053.
  Split each transformer layer's matrix multiplies across the GPUs of one
  machine, the tensor parallelism of section 5.
- **Sergeev and Del Balso, *Horovod: fast and easy distributed deep
  learning in TensorFlow* (2018)**: https://arxiv.org/abs/1802.05799.
  Brought the bandwidth-optimal ring all-reduce to deep learning training.

## Further reading

- Horace He, *Making Deep Learning Go Brrrr From First Principles*: https://horace.io/brrr_intro.html
- *How to Scale Your Model* (a book on TPUs, GPUs and parallelism for transformers): https://jax-ml.github.io/scaling-book/
- Williams, Waterman and Patterson, *Roofline* (2009): https://doi.org/10.1145/1498765.1498785
- Dao et al., *FlashAttention* (2022): https://arxiv.org/abs/2205.14135
- Micikevicius et al., *Mixed Precision Training* (2017): https://arxiv.org/abs/1710.03740
- Micikevicius et al., *FP8 Formats for Deep Learning* (2022): https://arxiv.org/abs/2209.05433
- PyTorch automatic mixed precision: https://pytorch.org/docs/stable/amp.html
- NVIDIA, *CUDA C++ Programming Guide* (how one GPU family organises cores and memory): https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from primer._show import banner, say, table, takeaway
from primer.ml.inference import HBM_BANDWIDTH, PEAK_FLOPS, kv_cache_bytes_per_token, weight_bytes

# ---------------------------------------------------------------------------
# 1. The workload: a matrix multiply, and how it splits across cores
# ---------------------------------------------------------------------------


def matmul_flops(m: int, n: int, k: int) -> int:
    """FLOPs to multiply an (m, k) matrix by a (k, n) matrix.

    Each of the m·n outputs is a dot product of length k: k multiplies and k
    adds, counted as 2 FLOPs per multiply-add (one fused instruction on a GPU).
    """
    return 2 * m * n * k


def counted_matmul(A: list[list[float]], B: list[list[float]]) -> tuple[list[list[float]], int, int]:
    """Multiply two small matrices with plain loops, counting every multiply and add.

    Slow on purpose: the loop *is* the definition, and the counters prove the
    2·m·n·k rule instead of asserting it.
    """
    m, k, n = len(A), len(B), len(B[0])
    C = [[0 for _ in range(n)] for _ in range(m)]
    multiplies = adds = 0
    for i in range(m):
        for j in range(n):
            # One output cell: its own dot product, needing nothing from any other cell.
            total = 0
            for p in range(k):
                product = A[i][p] * B[p][j]
                multiplies += 1
                total = total + product
                adds += 1
            C[i][j] = total
    return C, multiplies, adds


def parallel_rounds(m: int, n: int, k: int, cores: int) -> int:
    """Rounds of multiply-adds to finish an (m,k)@(k,n) multiply on `cores` identical cores.

    A deliberately simple model: each core takes whole output cells and works
    through a cell's k multiply-adds one per round. Cells are independent, so
    they share out perfectly until every cell has its own core; after that the
    extra cores have nothing to do and k rounds remain.
    """
    return math.ceil(m * n / cores) * k


# ---------------------------------------------------------------------------
# 2. The memory hierarchy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemoryLevel:
    """One level of storage, as seen from the arithmetic units of one GPU."""

    name: str
    capacity_bytes: float
    bandwidth_bytes_per_s: float
    picture: str  # the kitchen analogy used in the lesson


# Round, illustrative orders of magnitude for one datacenter GPU and its
# machine, not any product's specification. HBM uses the same bandwidth as
# primer.ml.inference so the two lessons describe the same imaginary chip.
MEMORY_HIERARCHY: list[MemoryLevel] = [
    MemoryLevel("registers", 20e6, 100e12, "the cook's hands"),
    MemoryLevel("on-chip SRAM", 50e6, 20e12, "the cutting board"),
    MemoryLevel("HBM", 80e9, HBM_BANDWIDTH, "the fridge in the kitchen"),
    MemoryLevel("host memory", 1e12, 50e9, "the storeroom down the hall"),
    MemoryLevel("local disk", 10e12, 10e9, "the warehouse across town"),
    MemoryLevel("network", 1e15, 50e9, "other restaurants' pantries, by courier"),
]


def level(name: str) -> MemoryLevel:
    """Look up a level of `MEMORY_HIERARCHY` by name."""
    return next(lv for lv in MEMORY_HIERARCHY if lv.name == name)


def transfer_seconds(nbytes: float, level_name: str) -> float:
    """Time to stream `nbytes` from one level at its full bandwidth (ignoring latency)."""
    return nbytes / level(level_name).bandwidth_bytes_per_s


# ---------------------------------------------------------------------------
# 3. Arithmetic intensity: a tiled matrix multiply that counts its traffic
# ---------------------------------------------------------------------------


@dataclass
class Traffic:
    """What a kernel moved between slow memory (HBM) and fast memory (SRAM), in elements."""

    reads: int = 0
    writes: int = 0
    flops: int = 0
    fast_memory_peak: int = 0


def tiled_matmul(A: np.ndarray, B: np.ndarray, tile: int) -> tuple[np.ndarray, Traffic]:
    """C = A @ B computed tile by tile, counting every element that crosses from slow to fast memory.

    For each (tile × tile) block of C: keep an accumulator in fast memory, and
    walk along the shared dimension loading one tile of A and one tile of B at
    a time. Each loaded element is then used `tile` times before it is thrown
    away, which is the reuse that cuts traffic. `tile=1` is no reuse at all:
    every multiply fetches both of its inputs.
    """
    m, k = A.shape
    k2, n = B.shape
    assert k == k2, "inner dimensions must match"
    assert m % tile == n % tile == k % tile == 0, "tiles must divide the matrix evenly"
    C = np.zeros((m, n))
    t = Traffic()
    for i0 in range(0, m, tile):
        for j0 in range(0, n, tile):
            acc = np.zeros((tile, tile))  # lives in fast memory for the whole walk along k
            for k0 in range(0, k, tile):
                a = A[i0:i0 + tile, k0:k0 + tile]  # load from slow memory
                b = B[k0:k0 + tile, j0:j0 + tile]
                t.reads += a.size + b.size
                t.fast_memory_peak = max(t.fast_memory_peak, a.size + b.size + acc.size)
                acc += a @ b  # tile³ multiply-adds on data already on chip
                t.flops += 2 * a.shape[0] * b.shape[1] * a.shape[1]
            C[i0:i0 + tile, j0:j0 + tile] = acc  # one write per output element, at the end
            t.writes += acc.size
    return C, t


def matmul_reads(n: int, tile: int) -> int:
    """Elements read from slow memory by `tiled_matmul` for two (n, n) matrices: 2n³ / tile."""
    return 2 * n**3 // tile


def matmul_intensity(n: int, tile: int, bytes_per_element: float) -> float:
    """FLOPs per byte of slow-memory traffic for an (n, n) tiled multiply, reads plus writes."""
    moved = (matmul_reads(n, tile) + n * n) * bytes_per_element
    return matmul_flops(n, n, n) / moved


# ---------------------------------------------------------------------------
# 4. Number formats, from scratch
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FloatFormat:
    """A binary floating-point format: 1 sign bit, some exponent bits, some mantissa bits.

    A normal value is (-1)^sign × 2^(exponent - bias) × (1 + mantissa / 2^M).
    `has_infinity` formats (IEEE style) reserve the all-ones exponent for
    infinity and NaN. The fp8 E4M3 format does not: it keeps that exponent
    for ordinary numbers and gives up only one pattern, to NaN, which buys it
    one more power of two of range.
    """

    name: str
    exponent_bits: int
    mantissa_bits: int
    has_infinity: bool = True

    @property
    def bits(self) -> int:
        return 1 + self.exponent_bits + self.mantissa_bits

    @property
    def bias(self) -> int:
        # Stored exponents are unsigned; the bias centres them so small and large numbers get equal room.
        return 2 ** (self.exponent_bits - 1) - 1

    @property
    def max_exponent_field(self) -> int:
        top = 2**self.exponent_bits - 1
        return top - 1 if self.has_infinity else top

    @property
    def max_mantissa_at_top(self) -> int:
        # E4M3 spends the all-ones mantissa at the top exponent on NaN.
        return 2**self.mantissa_bits - 1 if self.has_infinity else 2**self.mantissa_bits - 2

    @property
    def max_value(self) -> float:
        """The largest finite number the format can hold."""
        return (1 + self.max_mantissa_at_top / 2**self.mantissa_bits) * 2.0 ** (self.max_exponent_field - self.bias)

    @property
    def min_normal(self) -> float:
        """The smallest number held at full precision."""
        return 2.0 ** (1 - self.bias)

    @property
    def min_subnormal(self) -> float:
        """The smallest number above zero at all (with a single significant bit)."""
        return 2.0 ** (1 - self.bias - self.mantissa_bits)

    @property
    def epsilon(self) -> float:
        """The gap between 1 and the next number up: the format's relative precision."""
        return 2.0**-self.mantissa_bits


FP32 = FloatFormat("fp32", 8, 23)
BF16 = FloatFormat("bf16", 8, 7)
FP16 = FloatFormat("fp16", 5, 10)
FP8_E5M2 = FloatFormat("fp8 E5M2", 5, 2)
FP8_E4M3 = FloatFormat("fp8 E4M3", 4, 3, has_infinity=False)
FLOAT_FORMATS = [FP32, BF16, FP16, FP8_E5M2, FP8_E4M3]


def _overflow_fields(sign: int, fmt: FloatFormat) -> tuple[int, int, int]:
    """Fields for a value too big to hold: infinity if the format has it, otherwise its NaN."""
    if fmt.has_infinity:
        return sign, 2**fmt.exponent_bits - 1, 0
    return sign, 2**fmt.exponent_bits - 1, 2**fmt.mantissa_bits - 1


def encode(x: float, fmt: FloatFormat) -> tuple[int, int, int]:
    """Round `x` to the nearest value `fmt` can hold and return its (sign, exponent, mantissa) fields.

    Plain arithmetic, no bit tricks: find the power of two just below |x|,
    count how many steps of that binade's spacing |x| is, round to a whole
    number of steps (ties to even, like hardware), and split the step count
    into the hidden leading 1 and the stored mantissa.
    """
    M, bias = fmt.mantissa_bits, fmt.bias
    if math.isnan(x):
        return 0, 2**fmt.exponent_bits - 1, 2**M - 1
    sign = 1 if math.copysign(1.0, x) < 0 else 0
    a = abs(x)
    if a == 0.0:
        return sign, 0, 0
    if math.isinf(a):
        return _overflow_fields(sign, fmt)
    # frexp gives a = f · 2^e with f in [0.5, 1), so a lies in [2^(e-1), 2^e).
    exp = math.frexp(a)[1] - 1
    # Below the normal range the spacing stops shrinking: that is what subnormals are.
    exp = max(exp, 1 - bias)
    # a measured in steps of 2^(exp - M); Python's round() breaks ties to even, as hardware does.
    steps = round(a / 2.0 ** (exp - M))
    if steps == 2 ** (M + 1):
        # Rounding carried into the next power of two: 1.111… became 10.000….
        steps, exp = steps // 2, exp + 1
    if steps < 2**M:
        field, mantissa = 0, steps  # subnormal (or zero): no hidden leading 1
    else:
        field, mantissa = exp + bias, steps - 2**M  # the leading 1 is implied, not stored
    if field > fmt.max_exponent_field or (field == fmt.max_exponent_field and mantissa > fmt.max_mantissa_at_top):
        return _overflow_fields(sign, fmt)
    return sign, field, mantissa


def decode(sign: int, exponent: int, mantissa: int, fmt: FloatFormat) -> float:
    """The number a (sign, exponent, mantissa) triple stands for in `fmt`."""
    M, bias, top = fmt.mantissa_bits, fmt.bias, 2**fmt.exponent_bits - 1
    if fmt.has_infinity and exponent == top:
        magnitude = math.inf if mantissa == 0 else math.nan
    elif not fmt.has_infinity and exponent == top and mantissa == 2**M - 1:
        magnitude = math.nan
    elif exponent == 0:
        magnitude = mantissa * 2.0 ** (1 - bias - M)  # subnormal: 0.mantissa × 2^(1 - bias)
    else:
        magnitude = (2**M + mantissa) * 2.0 ** (exponent - bias - M)  # 1.mantissa × 2^(exponent - bias)
    return -magnitude if sign else magnitude


def round_to(x: float, fmt: FloatFormat) -> float:
    """`x` as it comes back after being stored in `fmt`."""
    return decode(*encode(x, fmt), fmt)


def bit_string(x: float, fmt: FloatFormat) -> str:
    """The stored bits of `x`, spaced as sign, exponent, mantissa."""
    s, e, m = encode(x, fmt)
    return f"{s} {e:0{fmt.exponent_bits}b} {m:0{fmt.mantissa_bits}b}"


def multiplier_cells(fmt: FloatFormat) -> int:
    """One-bit cells in a schoolbook (array) multiplier for the format's significands.

    Multiplying two p-bit numbers the long-multiplication way needs p × p
    one-bit products, with p = mantissa bits + the hidden leading 1. The
    exponents only need adding, which is cheap by comparison.
    """
    p = fmt.mantissa_bits + 1
    return p * p


# ---------------------------------------------------------------------------
# 5. Many GPUs: ring all-reduce and the cost of talking
# ---------------------------------------------------------------------------

# Illustrative, per GPU: a fast link between GPUs inside one machine, and the
# network card each GPU uses to reach GPUs in other machines (the "network"
# row of MEMORY_HIERARCHY).
IN_MACHINE_LINK = 500e9
BETWEEN_MACHINES_LINK = level("network").bandwidth_bytes_per_s


def ring_all_reduce(vectors: list[np.ndarray]) -> tuple[list[np.ndarray], list[int]]:
    """Sum one vector per GPU so every GPU ends with the total, passing chunks around a ring.

    Each GPU splits its vector into N chunks. Phase 1 (reduce-scatter): for
    N-1 steps, every GPU passes one chunk to its right-hand neighbour, which
    adds it to its own copy; afterwards GPU g owns the finished sum of chunk
    (g + 1) mod N. Phase 2 (all-gather): for N-1 more steps, the finished
    chunks travel round the ring and overwrite the stale copies.

    Returns every GPU's final vector and how many numbers each GPU sent.
    """
    N = len(vectors)
    chunks = [np.array_split(v.astype(float).copy(), N) for v in vectors]
    sent = [0] * N
    for step in range(N - 1):
        # All sends in a step happen at once, so read every outgoing chunk before anyone adds.
        outgoing = [(g, (g - step) % N, chunks[g][(g - step) % N].copy()) for g in range(N)]
        for g, c, data in outgoing:
            chunks[(g + 1) % N][c] += data
            sent[g] += data.size
    for step in range(N - 1):
        outgoing = [(g, (g + 1 - step) % N, chunks[g][(g + 1 - step) % N].copy()) for g in range(N)]
        for g, c, data in outgoing:
            chunks[(g + 1) % N][c] = data
            sent[g] += data.size
    return [np.concatenate(c) for c in chunks], sent


def all_reduce_seconds(nbytes: float, n_gpus: int, link_bytes_per_s: float) -> float:
    """Time for a ring all-reduce: each GPU sends 2(N-1)/N of the data over its link."""
    return 2 * (n_gpus - 1) / n_gpus * nbytes / link_bytes_per_s


def training_step_seconds(params: float, tokens: int, flops_per_second: float = PEAK_FLOPS) -> float:
    """Lower bound on one GPU's arithmetic for a training step: 6 FLOPs per parameter per token."""
    return 6 * params * tokens / flops_per_second


# ---------------------------------------------------------------------------
# 6. Serving: weights plus KV cache must fit
# ---------------------------------------------------------------------------

# Llama 3 70B's attention shape: 80 layers, 8 key/value heads of 128 dimensions.
LLAMA3_70B_SHAPE = dict(layers=80, kv_heads=8, head_dim=128)


def max_cache_tokens(gpu_bytes: float, params: float, weight_bits: int, kv_bits: int,
                     layers: int, kv_heads: int, head_dim: int) -> int:
    """Tokens of KV cache that fit beside the weights (0 if the weights alone don't fit).

    The formulas are primer.ml.inference's; this lesson varies the number format.
    """
    free = gpu_bytes - weight_bytes(params, weight_bits)
    if free <= 0:
        return 0
    return int(free // kv_cache_bytes_per_token(layers, kv_heads, head_dim, kv_bits))


# ---------------------------------------------------------------------------
# 7. Figures (rendered into the HTML docs by `make figures`)
# ---------------------------------------------------------------------------


def _relative_spacing(x: np.ndarray, fmt: FloatFormat) -> np.ndarray:
    """Gap to the next storable value, divided by x; NaN past the format's largest value."""
    exp = np.maximum(np.floor(np.log2(x)), 1 - fmt.bias)  # subnormals share the smallest normal spacing
    rel = 2.0 ** (exp - fmt.mantissa_bits) / x
    return np.where(x <= fmt.max_value, rel, np.nan)


def figures() -> dict:
    """Plot this lesson's data. matplotlib is imported here, and only here,
    so the lesson itself needs nothing beyond NumPy."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, RED, GREEN, ORANGE, MUTED = "#2563eb", "#dc2626", "#059669", "#d97706", "#9ca3af"
    figs = {}

    # --- 1. Parallel rounds: speed-up until the independent work runs out ----
    cores = 2 ** np.arange(0, 15)
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(cores, [parallel_rounds(64, 64, 64, int(c)) for c in cores], "o-", color=BLUE)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.axvline(64 * 64, color=MUTED, ls="--")
    ax.text(64 * 64 / 1.3, 1e4, "cores = cells\n(4,096)", color="#4b5563", ha="right")
    ax.set(xlabel="cores working at once", ylabel="rounds to finish",
           title="A 64 × 64 × 64 multiply: faster until every cell has a core")
    figs["parallel_rounds"] = fig

    # --- 2. The memory hierarchy: capacity up, bandwidth down --------------
    names = [lv.name for lv in MEMORY_HIERARCHY]
    y = np.arange(len(names))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    a1.barh(y, [lv.capacity_bytes for lv in MEMORY_HIERARCHY], color=BLUE)
    a2.barh(y, [lv.bandwidth_bytes_per_s for lv in MEMORY_HIERARCHY], color=ORANGE)
    a1.set_yticks(y, names)
    a1.set(xscale="log", xlabel="capacity (bytes)", title="Further out holds more...")
    a2.set(xscale="log", xlabel="bandwidth (bytes per second)", title="...and moves it more slowly")
    for a in (a1, a2):
        a.grid(axis="y", visible=False)
    fig.text(0.5, -0.02, "Illustrative orders of magnitude for one datacenter GPU, not a product specification",
             ha="center", color="#4b5563", fontsize=8)
    fig.tight_layout()
    figs["hierarchy"] = fig

    # --- 3. Tiling: traffic falls, intensity rises --------------------------
    n_sim = 32
    tiles = [1, 2, 4, 8, 16, 32]
    measured = [tiled_matmul(np.ones((n_sim, n_sim)), np.ones((n_sim, n_sim)), t)[1].reads for t in tiles]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4))
    a1.loglog(tiles, [matmul_reads(n_sim, t) for t in tiles], color=MUTED, base=2, label="formula 2n³/T")
    a1.loglog(tiles, measured, "o", color=BLUE, base=2, label="counted by tiled_matmul")
    a1.set(xlabel="tile width T", ylabel="numbers read from HBM", title=f"Reads for two {n_sim} × {n_sim} matrices")
    a1.legend(frameon=False)
    big = 4096
    ts = 2 ** np.arange(0, 12)
    for b, color, label in ((2, BLUE, "16-bit"), (1, GREEN, "8-bit")):
        a2.plot(ts, [matmul_intensity(big, int(t), b) for t in ts], "o-", color=color, label=label)
    a2.set_xscale("log", base=2)
    a2.set_yscale("log")
    ridge = PEAK_FLOPS / HBM_BANDWIDTH
    a2.axhline(ridge, color=RED, ls="--")
    a2.text(1.2, ridge * 1.25, f"break-even ≈ {ridge:.0f} FLOPs/byte", color=RED)
    a2.set(xlabel="tile width T", ylabel="FLOPs per byte moved", title=f"Intensity of a {big} × {big} multiply")
    a2.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    figs["tiling"] = fig

    # --- 4. Bit layouts, drawn to scale -------------------------------------
    rows = [(f.name, [("sign", 1), ("exponent", f.exponent_bits), ("mantissa", f.mantissa_bits)]) for f in FLOAT_FORMATS]
    rows += [("int8", [("integer", 8)]), ("int4", [("integer", 4)])]
    colors = {"sign": MUTED, "exponent": ORANGE, "mantissa": BLUE, "integer": GREEN}
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for r, (name, parts) in enumerate(rows):
        yy, x0 = len(rows) - 1 - r, 0
        for part, width in parts:
            ax.barh(yy, width, left=x0, height=0.7, color=colors[part], edgecolor="white", linewidth=0.5)
            if width >= 2:
                ax.text(x0 + width / 2, yy, str(width), ha="center", va="center", color="white", fontsize=8)
            x0 += width
        ax.text(x0 + 0.4, yy, f"{x0} bits", va="center", fontsize=8, color="#4b5563")
    ax.set_yticks(range(len(rows)), [name for name, _ in rows][::-1])
    ax.set_xlim(0, 36)
    ax.set_xlabel("bits")
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(handles, list(colors), frameon=False, loc="lower right")
    ax.set_title("Where each format spends its bits")
    figs["format_layouts"] = fig

    # --- 5. Relative precision across magnitudes ----------------------------
    x = np.logspace(-10, 6, 1200)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    for fmt, color in zip(FLOAT_FORMATS, (BLUE, GREEN, ORANGE, RED, "#7c3aed")):
        ax.loglog(x, _relative_spacing(x, fmt), color=color, label=fmt.name)
    for bits, ls in ((8, "--"), (4, ":")):
        step = 1 / (2 ** (bits - 1) - 1)  # scale chosen so 1.0 lands on the top code
        ax.loglog(x, np.where(x <= 1, step / x, np.nan), color="#4b5563", ls=ls, label=f"int{bits}, scale 1/{2 ** (bits - 1) - 1}")
    ax.set_ylim(1e-8, 10)
    ax.set(xlabel="size of the number stored", ylabel="gap to the next value ÷ the number",
           title="Relative precision: flat for floats, rising for integers")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    figs["precision"] = fig

    # --- 6. Communication vs. computation -----------------------------------
    gpus = 2 ** np.arange(1, 11)
    grad_bytes, params, tokens = 14e9, 7e9, 16_384
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.semilogx(gpus, [all_reduce_seconds(grad_bytes, int(g), BETWEEN_MACHINES_LINK) for g in gpus], "o-", color=RED,
                base=2, label="all-reduce over the network (50 GB/s)")
    ax.semilogx(gpus, [all_reduce_seconds(grad_bytes, int(g), IN_MACHINE_LINK) for g in gpus], "o-", color=BLUE,
                base=2, label="all-reduce over in-machine links (500 GB/s)")
    ax.axhline(training_step_seconds(params, tokens), color=MUTED, lw=3, label="arithmetic per step (16k tokens)")
    ax.set_ylim(0, 0.8)
    ax.set(xlabel="GPUs in the all-reduce", ylabel="seconds per training step",
           title="7B parameters, 16-bit gradients: talking vs. computing")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    figs["communication"] = fig

    # --- 7. Will it fit? ----------------------------------------------------
    gpu, params = 80e9, 70e9
    choices = [("16-bit", 16), ("fp8", 8), ("4-bit", 4)]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for i, (label, bits) in enumerate(choices):
        w = weight_bytes(params, bits) / 1e9
        free = max(gpu / 1e9 - w, 0)
        ax.bar(i, w, color=BLUE, label="weights" if i == 0 else None)
        ax.bar(i, free, bottom=w, color="#93c5fd", label="room for KV cache" if i == 0 else None)
        tokens_fit = max_cache_tokens(gpu, params, bits, 16, **LLAMA3_70B_SHAPE)
        note = "does not fit" if tokens_fit == 0 else f"{tokens_fit:,} tokens"
        ax.text(i, max(w, gpu / 1e9) + 3, note, ha="center", fontsize=9)
    ax.axhline(gpu / 1e9, color=RED, ls="--", label="80 GB of GPU memory")
    ax.set_xticks(range(len(choices)), [f"{label} weights" for label, _ in choices])
    ax.set_ylim(0, 185)
    ax.set(ylabel="GB", title="A 70B model on one GPU (16-bit KV cache)")
    ax.legend(frameon=False, loc="upper right")
    figs["serving_fit"] = fig

    return figs


# ---------------------------------------------------------------------------
# 8. Narrated walkthrough
# ---------------------------------------------------------------------------


def demo() -> None:
    banner("1. The workload: a matrix multiply, counted by hand")
    C, muls, adds = counted_matmul([[1, 2, 3], [4, 5, 6]], [[7, 8], [9, 10], [11, 12]])
    say(f"[[1,2,3],[4,5,6]] times [[7,8],[9,10],[11,12]] = {C}.")
    say(
        f"""
        That took {muls} multiplies and {adds} adds: 2·m·n·k = {matmul_flops(2, 2, 3)} FLOPs.
        No output cell needed another cell's answer, so they could all be
        computed at once. A 4096-cube multiply is {matmul_flops(4096, 4096, 4096):.3e} FLOPs.
        """
    )
    table(["cores", "rounds for a 64 × 64 × 64 multiply"], [(c, parallel_rounds(64, 64, 64, c)) for c in (1, 64, 4096, 8192)])
    takeaway("A GPU wins by doing thousands of independent multiply-adds at once; it needs that much independent work.")

    banner("2. The memory hierarchy")
    table(
        ["level", "holds", "moves per second", "ms to stream 1 GB", "picture"],
        [(lv.name, f"{lv.capacity_bytes:.0e} B", f"{lv.bandwidth_bytes_per_s:.2e} B", transfer_seconds(1e9, lv.name) * 1e3, lv.picture)
         for lv in MEMORY_HIERARCHY],
        floatfmt=".2f",
    )
    say(
        """
        Illustrative round numbers. Reading 16 GB of weights from HBM takes
        4.78 ms; from host memory it would take 320 ms.
        """
    )
    takeaway("Each step away from the arithmetic is bigger and slower; full speed means living in HBM or closer.")

    banner("3. Tiling: same FLOPs, far fewer bytes")
    rows = []
    for t in (1, 2, 4):
        _, traffic = tiled_matmul(np.ones((4, 4)), np.ones((4, 4)), t)
        rows.append((f"{t} × {t}", traffic.reads, traffic.writes, traffic.flops, traffic.flops / (traffic.reads + traffic.writes)))
    table(["tile", "reads", "writes", "FLOPs", "FLOPs per number moved"], rows, floatfmt=".2f")
    say(
        f"""
        For 4096 × 4096 matrices with 128-wide tiles: {matmul_intensity(4096, 128, 2):.1f} FLOPs per byte
        in 16-bit, {matmul_intensity(4096, 128, 1):.1f} in 8-bit. The chip breaks even at
        {PEAK_FLOPS / HBM_BANDWIDTH:.0f}. FlashAttention, kernel fusion and batching all
        raise this ratio.
        """
    )
    takeaway("Speed is usually set by bytes moved, not FLOPs; reuse every byte you fetch as often as you can.")

    banner("4. Number formats, built from scratch")
    say(f"-6.5 in bf16 is stored as {bit_string(-6.5, BF16)} (sign, exponent, mantissa).")
    table(
        ["format", "bits", "largest", "smallest normal", "gap above 1", "0.1 is stored as", "multiplier cells"],
        [(f.name, f.bits, f"{f.max_value:.4g}", f"{f.min_normal:.3g}", f"{f.epsilon:.3g}", f"{round_to(0.1, f):.8g}", multiplier_cells(f))
         for f in FLOAT_FORMATS],
    )
    say(
        f"""
        Range: 1e-8 in fp16 becomes {round_to(1e-8, FP16)}; in bf16 it is {round_to(1e-8, BF16):.4g}.
        Precision: 1 + 0.001 in bf16 is {round_to(1.001, BF16)}; in fp32 it is {round_to(1.001, FP32):.7g}.
        Overflow: 70,000 in fp16 is {round_to(70_000.0, FP16)}; 500 in fp8 E4M3 is {round_to(500.0, FP8_E4M3)}.
        """
    )
    takeaway("Exponent bits buy range, mantissa bits buy precision; fewer bits buy speed three ways.")

    banner("5. Many GPUs: ring all-reduce")
    results, sent = ring_all_reduce([np.arange(8.0) * (g + 1) for g in range(4)])
    say(f"Four GPUs, 8 numbers each. After the ring, every GPU holds {results[0].tolist()}; each sent {sent[0]} numbers.")
    table(
        ["link", "all-reduce of 14 GB on 8 GPUs (ms)", "arithmetic per step (ms)"],
        [("in one machine", all_reduce_seconds(14e9, 8, IN_MACHINE_LINK) * 1e3, training_step_seconds(7e9, 16_384) * 1e3),
         ("between machines", all_reduce_seconds(14e9, 8, BETWEEN_MACHINES_LINK) * 1e3, training_step_seconds(7e9, 16_384) * 1e3)],
        floatfmt=".0f",
    )
    takeaway("Put chatty parallelism on fast links; send the once-per-step traffic over the network.")

    banner("6. Will a 70B model fit on one 80 GB GPU?")
    table(
        ["weights", "weight GB", "tokens of 16-bit cache", "tokens of 8-bit cache"],
        [(f"{b}-bit", weight_bytes(70e9, b) / 1e9, max_cache_tokens(80e9, 70e9, b, 16, **LLAMA3_70B_SHAPE),
          max_cache_tokens(80e9, 70e9, b, 8, **LLAMA3_70B_SHAPE)) for b in (16, 8, 4)],
        floatfmt=".0f",
    )
    takeaway("The number format decides whether a model fits, how many users share a GPU, and how fast each token comes.")


if __name__ == "__main__":
    demo()
