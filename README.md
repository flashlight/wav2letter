# Training with speaker information
Several changes needed in order to train with speaker information (All the details apply for librispeech dataset at the moment).

## Arguments
* Provide the suffix of the speaker files using the `-speaker` argument, the default values for librispeech is  `spk`.
* Provide the path to the list of the speakers (the same as in the the letters/phones list), using the `-speakerslist` argument.

## Architecture
In order to add another branch to the network you need to add the desire architecture of the new branch AFTER the regular architecture.
One special line separates between the two architectures: `CT <layer number>`. This new line will create a ConcatTable from the given layer number and use the rest of the architecture file as the architecture for the new branch.

The last layer can contains the NSPEAERS token, and then the model automatically adds the number of speakers given by the speakers list.
Another two special layer that one can use are the:
*`GRL <float number>`: This new line will create GradientReverse layer with the given lambda value.
*`LSE`: This new line will create LogSumExp over time.

For example:
'
regular arch file...
...
CT 9
GRL 0.00001
C 220 484 15 1
GLU
DO 0.5
C 242 NSPEAERS 1 1
LSE
'
