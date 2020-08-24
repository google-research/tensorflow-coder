# TensorFlow Coder (TF-Coder)

TF-Coder is a _program synthesis_ tool that helps you write TensorFlow code.
First, the tool asks for an input-output example of the desired tensor
transformation. Then, it runs a combinatorial search to find TensorFlow
expressions that perform that transformation. TF-Coder’s output is real
TensorFlow code that you can include in your projects.

## Quick Links

[**Try TF-Coder!**](https://colab.research.google.com/github/google-research/tensorflow-coder/blob/master/TF-Coder_Colab.ipynb)

The TF-Coder tool is ready-to-use at this link. Everything is already packaged
together in a Colab notebook, so no installation or download is needed.

For more information about TF-Coder, see the following documents:
* [**TF-Coder Tutorial**](Tutorial.md): walks you through using TF-Coder to
  solve tensor manipulation tasks, and provides tips on getting the most out of
  TF-Coder.
* [**User Journeys**](UserJourneys.md): illustrates several realistic scenarios
  where TF-Coder can help accelerate your TensorFlow development in different
  ways.

## Contents

* [What is TF-Coder?](#what-is-tf-coder)
* [Tutorial and Further Reading](#tutorial-and-further-reading)
* [Optional: Using TF-Coder Outside Colab](#optional-using-tf-coder-outside-colab)
* [Citation](#citation)

## What is TF-Coder?

When manipulating tensors, one must keep track of multiple dimensions, tensor
shape and DType compatibility, and of course mathematical correctness.
Additionally, there are hundreds of TensorFlow operations, and finding the right
ones to use can be a challenge.

TensorFlow Coder, or TF-Coder, can help you write tricky tensor manipulations in
TensorFlow. Instead of coding your tensor manipulation directly, you can just
demonstrate it through an illustrative input-output example, and TF-Coder can
produce the corresponding code automatically. TF-Coder performs an efficient
combinatorial search over compositions of TensorFlow operations, until it finds
a TensorFlow expression that matches the given input-output example.

TF-Coder allows you to:
* Program in TensorFlow by example
* Find the right function to use
* Automatically combine functions in clever ways
* Spend less time debugging

TF-Coder is primarily a development tool for TensorFlow users. If you just want
to use TF-Coder as a tool, you don’t need to install anything, as the tool is
ready-to-use in this
[Colab notebook](https://colab.research.google.com/github/google-research/tensorflow-coder/blob/master/TF-Coder_Colab.ipynb).

### Caveats

There are limitations to TF-Coder. It can currently find solutions involving 3-4
operations within a minute of searching, but solutions involving 6 or more
operations are too complex to find in a reasonable amount of time. Furthermore,
TF-Coder currently does not support complex or string tensors, or RaggedTensors.
The full list of supported operations can be found in the
[Colab notebook](https://colab.research.google.com/github/google-research/tensorflow-coder/blob/master/TF-Coder_Colab.ipynb#scrollTo=Q6uRr4x9WHRC).

In addition, TF-Coder only guarantees that its solutions work for the given
input-output example. The tool searches for a simple TensorFlow expression that
matches the provided input-output example, but sometimes this solution is too
simple and doesn’t generalize in the intended way. It can be helpful to make the
example as unambiguous as possible, which can often be achieved by adding more
numbers to the input and output tensors. Please review TF-Coder’s solutions to
ensure that they correctly implement the intended behavior.

In the Colab tool, we would like to log the problems given to TF-Coder and the
resulting solutions, so that we can improve the tool and build a dataset that
will accelerate program synthesis research in general, but this data collection
is completely optional.

## Tutorial and Further Reading

For more information about TF-Coder, see the following documents:
* [**TF-Coder Tutorial**](Tutorial.md): walks you through using TF-Coder to
  solve tensor manipulation tasks, and provides tips on getting the most out of
  TF-Coder.
* [**User Journeys**](UserJourneys.md): illustrates several realistic scenarios
  where TF-Coder can help accelerate your TensorFlow development in different
  ways.
* [**Our research paper**](https://arxiv.org/abs/2003.09040): describes the
  technology behind TF-Coder.

## Optional: Using TF-Coder Outside Colab

Because TF-Coder is primarily a development tool and not a library that you use
in your code, we hope that the provided
[Colab notebook](https://colab.research.google.com/github/google-research/tensorflow-coder/blob/master/TF-Coder_Colab.ipynb)
is sufficient for your use cases.

However, if you would rather not use the Colab notebook, you can still install
TF-Coder as a Python package yourself:
```
pip install --user tf-coder
```

To run the TF-Coder search as a library, follow the code example in
[`tf_coder_main.py`](tf_coder/tf_coder_main.py).

To run TF-Coder on our benchmarks, run:
```
python3 tf_coder/value_search/value_search_main.py
```

To run tests, clone the repository and run `pytest`.

## Citation

If you find TF-Coder helpful for a research project, you may cite our [research
paper](https://arxiv.org/abs/2003.09040) as follows:
```
@article{TFCoder,
    title={{TF-Coder}: Program Synthesis for Tensor Manipulations},
    author={Kensen Shi and David Bieber and Rishabh Singh},
    year={2020},
    url={https://arxiv.org/abs/2003.09040},
    archivePrefix={arXiv},
    eprint={2003.09040}
}
```

## Disclaimer

This is a research project, not an official Google product.

To report a bug or make a feature request, please raise a
[GitHub issue](https://github.com/google-research/tensorflow-coder/issues).
