# TF-Coder User Journeys

In this document, we illustrate various scenarios where TF-Coder can help you
write TensorFlow code. As these scenarios show, we believe that TF-Coder can be
a useful tool, regardless of how much TensorFlow experience you already have. We
hope that you will consider using TF-Coder when encountering similar scenarios
in your own work.

TF-Coder allows you to:
* [Program in TensorFlow by example](#programming-in-tensorflow-by-example)
* [Find the right function to use](#tf-coder-helps-you-find-the-right-function-to-use)
* [Automatically combine functions in clever ways](#tf-coder-helps-you-combine-functions-in-clever-ways)
* [Spend less time debugging](#tf-coder-helps-you-write-correct-code-with-less-debugging)

## Programming in TensorFlow by example

Suppose you want to "add" an _M_-element vector with an _N_-element vector in a
broadcasted way to produce an _M_ x _N_ matrix containing all pairwise sums.
Instead of digging through TensorFlow documentation to figure out how to do
this, you can instead provide an input-output example (using _M_ = 3 and
_N_ = 4):

Input tensors, as a dict mapping input variable names to example tensor values:
```
inputs = {
    'rows': [10, 20, 30],
    'cols': [1, 2, 3, 4],
}
```

The desired output tensor, corresponding to the provided input tensors:
```
output = [[11, 12, 13, 14],
          [21, 22, 23, 24],
          [31, 32, 33, 34]]
```

Given this information (already entered into the TF-Coder Colab by default), the
TF-Coder tool will find the appropriate TensorFlow code automatically in a
fraction of a second:

<pre><b>tf.add(cols, tf.expand_dims(rows, 1))</b></pre>

The above problem was pretty simple just to illustrate the idea of programming
by example. TF-Coder can be useful for harder problems as well, as we'll see
below.

## TF-Coder helps you find the right function to use

Let's suppose you are working with a numerical feature such as the price of an
item. The prices in your dataset have a wide range, e.g., from under $10 to over
$1000. If these prices are used directly as features, your model may overfit to
specific prices in the training data, and it may also have difficulty with
outlier prices during evaluation.

To deal with these issues, you may want to use
[_bucketing_](https://developers.google.com/machine-learning/data-prep/transform/bucketing)
to transform the numerical prices into categorical features. For example, using
bucket boundaries of `[10, 50, 100, 1000]` means that prices under $10 should
fall into bucket `0`, prices between $10 and $50 fall into bucket `1`, and so
on.

After choosing bucket boundaries, how do you actually map the numerical prices
to the bucket indices using TensorFlow? For example, given the following bucket
boundaries and item prices:

```
# Input tensors
boundaries = [10, 50, 100, 1000]
prices = [15, 3, 50, 90, 100, 1001]
```

you want to compute the bucket number for each item:

```
# Output tensor
bucketed_prices = [1, 0, 2, 2, 3, 4]
```

Although TensorFlow comes with various bucketing operations, it may be tricky to
figure out which specific operation does this exact kind of bucketing. Since
TF-Coder can identify hundreds of Tensor operations by behavior, you can look up
the correct operation by providing an input-output example:

```
# Input-output example
inputs = {
    'boundaries': [10, 50, 100, 1000],
    'prices': [15, 3, 50, 90, 100, 1001],
}
output = [1, 0, 2, 2, 3, 4]
```

Within seconds, TF-Coder outputs the following solution:

<pre><b>tf.searchsorted(boundaries, prices, side='right')</b></pre>

This gives us a useful hint, and the documentation for
[`tf.searchsorted`](https://www.tensorflow.org/api_docs/python/tf/searchsorted)
confirms that this code indeed performs the bucketing as desired.

## TF-Coder helps you combine functions in clever ways

Now let's consider another problem: compute a 0-1 tensor that identifies the
maximum element of each row of the input tensor.

```
# Input tensor
scores = [[0.7, 0.2, 0.1],
          [0.4, 0.5, 0.1],
          [0.4, 0.4, 0.2],
          [0.3, 0.4, 0.3],
          [0.0, 0.0, 1.0]]

# Output tensor
top_scores = [[1, 0, 0],
              [0, 1, 0],
              [1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]]
```

Note that if the same largest element appears multiple times within a row, such
as in the third row of scores, then only the first such largest element should
be marked, so that every row of `top_scores` has exactly one entry of `1`.

Unlike in the last problem, there is no single TensorFlow function that performs
this computation. If you search the documentation for "max", you may find that
`tf.reduce_max`, `tf.argmax`, and `tf.maximum` are relevant, but which one
should you use? `tf.reduce_max` produces `[0.7, 0.5, 0.4, 0.4, 1.0]`,
`tf.argmax` produces `[0, 1, 0, 1, 2]`, and `tf.maximum` isn't right because it
takes two arguments. None of these look close to our desired output.

TF-Coder can help solve tricky problems like this. You can write the problem in
the form of an input-output example:

```
# Input-output example
inputs = {
    'scores': [[0.7, 0.2, 0.1],
               [0.4, 0.5, 0.1],
               [0.4, 0.4, 0.2],
               [0.3, 0.4, 0.3],
               [0.0, 0.0, 1.0]],
}
output = [[1, 0, 0],
          [0, 1, 0],
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]]
```

TF-Coder uses a combination of
[`tf.one_hot`](https://www.tensorflow.org/api_docs/python/tf/one_hot) and
[`tf.argmax`](https://www.tensorflow.org/api_docs/python/tf/math/argmax) in a
short solution to this problem:

<pre><b>tf.cast(tf.one_hot(tf.argmax(scores, axis=1), 3), tf.int32)</b></pre>

Through a detailed search over combinations of TensorFlow operations, TF-Coder
often finds elegant solutions like this, which may simplify and speed up your
TensorFlow programs.

## TF-Coder helps you write correct code with less debugging

Consider normalizing lists of integer counts into probability distributions by
dividing each row by the sum of that row. For instance:

```
# Input tensor
counts = [[0, 1, 0, 0],
          [0, 1, 1, 0],
          [1, 1, 1, 1]]

# Output tensor
normalized = [[0.0, 1.0, 0.0, 0.0],
              [0.0, 0.5, 0.5, 0.0],
              [0.25, 0.25, 0.25, 0.25]]
```

Even if you know relevant functions to use (`tf.reduce_sum` followed by
`tf.divide`), writing the correct code is still nontrivial. A first attempt may
look like this:

```
# First attempt
normalized = tf.divide(counts, tf.reduce_sum(counts, axis=1))
```

Is this right? There are many potential pitfalls to think about:
* Is the summation axis correct, or should it be `axis=0`?
* Are the shapes of `counts` and `tf.reduce_sum(counts, axis=1)` compatible for
  division, or do you need to reshape or transpose either of these?
* `counts` and `tf.reduce_sum(counts, axis=1)` are both `tf.int32` tensors. Can
  `tf.int32` tensors be divided, or do you need to cast them to a float DType
  first?
* Are the two arguments in the correct order, or should they be swapped?
* Does the output have type `tf.int32`, `tf.float32`, or something else?
* Is there a simpler or better way that was not considered?

You can give this task to TF-Coder with the following input-output example:

```
# Input-output example
inputs = {
    'counts': [[0, 1, 0, 0],
               [0, 1, 1, 0],
               [1, 1, 1, 1]],
}
output = [[0.0, 1.0, 0.0, 0.0],
          [0.0, 0.5, 0.5, 0.0],
          [0.25, 0.25, 0.25, 0.25]]
```

TF-Coder's solution is:

<pre><b>tf.cast(tf.divide(counts, tf.expand_dims(tf.reduce_sum(counts, axis=1), axis=1)), tf.float32)</b></pre>

By using TF-Coder to solve this problem, the mental burden of the exercise is
reduced. When TF-Coder produces the solution above, it is guaranteed that the
code correctly produces the example output when run on the example input.
TF-Coder's solution will also avoid any unnecessary steps. Thus, you can quickly
deduce the answers to most of the questions above: an extra `tf.expand_dims`
step is needed to make the shapes compatible for division, and the result of
`tf.divide` must be cast to `tf.float32` (in fact `tf.divide` returns a
`tf.float64` tensor when dividing two `tf.int32` tensors). In this way, TF-Coder
helps you write simple and correct code without painful debugging cycles.
