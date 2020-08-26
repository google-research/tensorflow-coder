# TF-Coder Tutorial

This is a tutorial for TF-Coder, a program synthesis tool for TensorFlow.

## Colab Basics

If you have used Colab before, feel free to skip this section.

* Colab is an interactive Python notebook environment, similar to Jupyter
  notebooks. A Colab notebook consists of code cells and text cells.
* After typing code into a code cell, press Shift-Enter, or click the play
  button in the top left corner of the cell, to execute the code.
* While a cell is running, the play button turns into a stop button. You can use
  this to interrupt execution of the code.
* Variables persist between cell executions.
* You can create new code cells by hovering between existing cells and clicking
  "+ Code".
* In the TF-Coder Colab, any edits you make are only visible to you and will be
  lost when you reload the page, unless you save a copy of the notebook. (If we
  later make improvements to the Colab notebook, they will not appear in your
  copy.)

## Setting up TF-Coder

This only needs to be done once per session of using TF-Coder.

1. Open the [TF-Coder Colab](https://colab.research.google.com/github/google-research/tensorflow-coder/blob/master/TF-Coder_Colab.ipynb).
2. Connect to a runtime by clicking the "Connect" button in the top right corner
   of the notebook.
3. Read the text in Step 0 of the Colab notebook. As explained in the cell, we
   would like to record the problems given to TF-Coder, as well as TF-Coder's
   solutions, in order to improve the tool and accelerate further research in
   program synthesis. This data collection is completely optional. Check or
   uncheck the boxes as desired, and then run the cell.
4. Run the cell in Step 1 of the Colab notebook to finish setup. This will take
   about 10 seconds to complete.

## Using TF-Coder

After performing the setup steps above, we are now ready to use the TF-Coder
tool to solve problems.

As a very simple example, suppose you want to "add" an _M_-element vector `rows`
with an _N_-element vector `cols` in a broadcasted way to produce an _M_ x _N_
matrix containing all pairwise sums. For example, using _M_ = 3 and _N_ = 4:

```
rows = [10, 20, 30]
cols = [1, 2, 3, 4]

output = [[11, 12, 13, 14],
          [21, 22, 23, 24],
          [31, 32, 33, 34]]
```

To use TF-Coder on this problem, the first step is to define the variables
`inputs`, `output`, `constants`, and `description`, which will then be used by
the TF-Coder tool. For example:

```
# A dict mapping input variable names to input tensors.
inputs = {
    'rows': [10, 20, 30],
    'cols': [1, 2, 3, 4],
}

# The corresponding output tensor.
output = [[11, 12, 13, 14],
          [21, 22, 23, 24],
          [31, 32, 33, 34]]

# A list of relevant scalar constants, if any.
constants = []

# An English description of the tensor manipulation.
description = 'add two vectors with broadcasting to get a matrix'
```

This is already entered in the TF-Coder Colab notebook (Step 2) by default.

* `inputs` is a dictionary that maps variable names to their tensor values.
  Tensors can be provided as lists (possibly multidimensional) or `tf.Tensor`
  objects.
  * If `inputs` is a list instead, TF-Coder will treat every element of
    that list as a separate tensor input, using the variable names `in1`, `in2`,
    etc.

* `output` is a single tensor, showing the result of the desired transformation
  when applied to the input tensors. `output` must stay in sync with `inputs` --
  if you modify the example input tensors, be sure to make corresponding
  modifications to `output` as well!

* `constants` is a list of scalar constants, if any, that are needed to solve
  the problem. TF-Coder uses heuristics to automatically identify some
  constants, but providing important constants explicitly can speed up the
  search.

* `description` is an English description of the tensor manipulation. This helps
  TF-Coder identify TensorFlow operations that are more relevant to the
  particular problem. However, having a good description is less important than
  having a correct and informative input-output example, and ensuring that
  necessary constants are provided.

After specifying the problem in Step 2 of the Colab notebook, using the above
format, don't forget to **run the cell**.

Then, run the cell in Step 3 of the Colab notebook. TF-Coder first prints out
the input and output tensors (as `tf.Tensor` objects, showing their shapes and
DTypes), constants (including those chosen automatically by heuristics), and
description. Then, TF-Coder will search for solutions, printing them out as they
are found.

Finally, it is important to verify that TF-Coder's solution actually implements
the desired transformation. In this case, TF-Coder produces a correct solution:

<pre><b>tf.add(cols, tf.expand_dims(rows, 1))</b></pre>

However, sometimes TF-Coder produces a solution that works for the given
input-output example, but does not generalize fully. These solutions are called
_false positives_.

### Dealing with False Positive Solutions

Consider another tensor manipulation problem: given a 1D tensor data, extract
all values from data that are greater than `5`, preserving their order.

As a first attempt, we can run TF-Coder using the following problem
specification:

```
inputs = {
    'data': [3, 2, 8, 6, 4],
}
output = [8, 6]
constants = [5]
description = 'extract values greater than 5'
```

Note that we included the constant `5` because we suspect that it is necessary to
solve this problem correctly.

At first glance, this input-output example seems clear -- the output contains
`8` and `7` because those are the only values greater than `5`. However,
TF-Coder produces the following solution:

```
data[2:-1]
```

Although this code produces the right answer for this input-output example, it
is clearly not the intended computation. This illustrates how an input-output
example that is clear to humans might still be ambiguous to TF-Coder, which is
fundamentally an efficient enumerative search over TensorFlow expressions. Thus,
it is important to manually check TF-Coder's solutions for correctness.

The false positive solution took advantage of an unintentional pattern in the
input-output example, where the values to extract were all consecutive. We can
modify the input-output example to eliminate this pattern. There are many ways
of doing this, including:

```
inputs = {
    'data': [3, 2, 8, 6, 4, 7]
}
output = [8, 6, 7]
constants = [5]
description = 'extract values greater than 5'
```

TF-Coder's old solution is no longer valid for the updated input-output example,
and TF-Coder produces a correct solution instead:

<pre><b>tf.boolean_mask(data, tf.greater(data, tf.constant(5)))</b></pre>

In general, we find that including more numbers in the input and output tensors
is an effective way to avoid false positive solutions. We discuss other ways of
avoiding false positives in the section below.

## Tips for Using TF-Coder

### General

* If TF-Coder finds a solution, it is _guaranteed_ that the solution produces
  the example output when run on the example inputs. However, it is _not
  guaranteed_ that the solution generalizes in the way you intend! Please
  carefully review solutions produced by TF-Coder before using them in your real
  project.

* TF-Coder will often produce a solution that uses hardcoded constants for
  shapes or lengths, e.g., `tf.reshape(to_flatten, (6,))` in order to flatten an
  input tensor with shape `(2, 3)`. You may need to manually change these
  constants to improve the generality of the solution, e.g., replacing `6` with
  `-1` in this case. Use the shape attribute to obtain dimension lengths of
  input tensors, e.g., `to_flatten.shape[0]` would be `2`.

* If you want to play with TensorFlow in Colab (e.g., to understand how a
  TF-Coder solution works or to test your own solution):
  * The TF-Coder Colab already imports TensorFlow 2 and Numpy, for your
    convenience.
  * Use `tf.constant` to create a tensor from the list format:
    ```
    >>> tf.constant([[13, 22], [17, 5]])
    <tf.Tensor: id=1, shape=(2, 2), dtype=int32, numpy=
    array([[13, 22],
           [17,  5]], dtype=int32)>

    >>> tf.constant(12.3)
    <tf.Tensor: id=2, shape=(), dtype=float32, numpy=12.3>
    ```
  * A Colab notebook can only have one cell running at a time. If you want to
    experiment with TensorFlow code while TF-Coder is running, consider doing so
    in a separate Python shell.

* TF-Coder's running time is exponential in the complexity of the solution.
  _Simplifying the problem_, or _breaking it down into multiple steps_, can help
  TF-Coder find solutions quickly. For instance, if you know that a reshape,
  transpose, cast, or other similar operation should be applied to an input or
  as the last operation to produce the output, consider applying that operation
  manually to the input-output example, to help TF-Coder focus on the more
  difficult parts.

### Input-Output Example

Creating a good input-output example is crucial for TF-Coder to find the
solution you want. The example should be robust enough to rule out _false
positive solutions_, which are TensorFlow expressions that work on the given
example, but fail to generalize in the desired way.

Here are some techniques that reduce the risk of false positives:

* **Include more numbers** in the input and output tensors. TF-Coder will only
  output a solution if it works on the provided example, so having many numbers
  in the output tensor means it is less likely for incorrect solutions to
  produce all of the correct numbers by chance.

* **Use random-looking numbers** in the input tensors. For example,
  `[18, 73, 34, 51]` would be a better input tensor than `[1, 2, 3, 4]`, since
  the former is not all consecutive and not all increasing. This helps eliminate
  patterns in the input tensors that false positive solutions can take advantage
  of.

* **Remove patterns from the output other than the intended one**. For example,
  if the output tensor is a selection of numbers from input tensors, make sure
  the selected numbers aren't all the maximum element along some axis, unless
  that is the intended pattern.

* **Include edge cases** where relevant. These could include negative numbers,
  zero, or duplicate numbers, when applicable to the problem.

* **Distinguish between indices and non-indices**. If you know a number should
  not be used as an index, consider making it out of range of valid indices
  (negative, too large, or even floating-point).

* **Follow any constraints that exist in your real program**. For example, if an
  input tensor only contains positive numbers, TF-Coder may produce a solution
  that doesn't generalize to negative numbers. Whether this is acceptable
  depends on whether that tensor could possibly contain negative numbers in your
  real program. Of course, depending on the problem, a completely general
  solution may be unnecessarily harder to find.

In general, false positive solutions are more common if the output tensor
contains a relatively low amount of information given the inputs. This may
happen if the output is a scalar or boolean tensor, or if the output is
constructed by selecting one or a few elements from an input. When possible, try
to include many numbers in the output so that it contains enough information to
unambiguously identify the intended transformation.

### Constants

* TF-Coder will print out the list of constants that it is using, including
  constants chosen through heuristics. This list is ordered with
  highest-priority constants at the beginning.
* If the intended solution requires a constant that is not in TF-Coder's printed
  list of constants, then TF-Coder will be _unable_ to find the intended
  solution. So, it is important to provide any necessary constants.
* If you explicitly provide constants, they will be used with the highest
  priority. Thus, even if TF-Coder's heuristics choose your desired constant, it
  may be better to provide the constant explicitly so that TF-Coder is more
  confident about using your constant.
* Providing extraneous constants will slow down the tool.

### Description

* The description is optional. If provided, it is used to prioritize TensorFlow
  operations that fit with the description.
* If you know of a TensorFlow operation (e.g., `tf.reduce_max`) that is
  relevant, include its name (e.g., "tf.reduce_max") anywhere in the
  description. This will lead TF-Coder to prioritize that operation.
* If possible, try to describe how the output should be computed, rather than
  what the output conceptually represents.
* A good description is less important than a good input-output example.

### Other Details and Advanced Options

* When running TF-Coder, you can set the time limit, the number of solutions to
  find, and whether solutions are required to use inputs.
  * Time limit: This is the maximum amount of time, in seconds, that TF-Coder
    will spend on the problem before giving up. Note that you can stop the tool
    at any time by pressing the cell's stop button.
  * Number of solutions: TF-Coder can continue searching for more solutions
    after the first solution is found. This can help you examine different ways
    of solving the problem. However, enabling multiple solutions will cause the
    entire search to slow down, even for the first solution.
  * Solution requirement: By default, solutions are required to use every input
    tensor at least once. This constraint can be relaxed to allow solutions that
    use only one input (if there are multiple inputs), or even solutions that
    use no inputs at all.

* By default, integer tensors have a DType of `tf.int32`, and float tensors have
  a DType of `tf.float32`. To specify a different DType, provide a `tf.Tensor`
  object instead of a list. For example:
  * If an input is given as `[3, 1, 7, 4]`, then it will have a DType of
    `tf.int32`.
  * If an input is given as `tf.constant([3, 1, 7, 4], dtype=tf.int64)`, then it
    will have a DType of `tf.int64`.

* A primitive scalar input can be specified with a Python float or int, and a
  scalar tensor can be specified with a `tf.Tensor`:
  * If an input is given as `[123]`, then it will be a 1-dimensional tensor with
    shape `(1,)`, equivalent to `tf.constant([123])`.
  * If an input is given as `123`, then it will remain a Python primitive int,
    not a `tf.Tensor`.
  * If an input is given as `tf.constant(123)`, then it will be a 0-dimensional
    scalar tensor with shape `()`.

* Input and output tensors can have at most 4 dimensions.
