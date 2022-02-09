### Brain-optimized neural network for auditory processing

The repository provides code that supports the results reported here: 

[Brain-optimized extraction of complex sound features that drive continuous auditory perception](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007992)<br /> by Berezutskaya J., Freudenburg Z.V., Güçlü U., van Gerven M.A.J. & Ramsey N.F. in *PLOS Computational Biology*, 2020


In the paper we report training of a data-driven neural model of auditory perception, with a minimum of theoretical assumptions about the relevant sound features. We show that it could provide an alternative approach and possibly a better match to the neural responses.

The repository contains

- the model specification using [Chainer](https://chainer.org/) 6.1.0
- the training routine
- the weights of one of the best performing models


![Alt text](/model.png?raw=true "Model architecture")

### Citation

If this code has been useful for you, please cite the related paper:
```
@article{berezutskaya2020brain,
  title={Brain-optimized extraction of complex sound features that drive continuous auditory perception},
  author={Berezutskaya, Julia and Freudenburg, Zachary V and G{\"u}{\c{c}}l{\"u}, Umut and van Gerven, Marcel AJ and Ramsey, Nick F},
  journal={PLoS computational biology},
  volume={16},
  number={7},
  pages={e1007992},
  year={2020},
  publisher={Public Library of Science San Francisco, CA USA}
}
```
