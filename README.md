# Training with speaker information
Several changes needed in order to train with speaker information (All the details apply for librispeech dataset at the moment).

## Special layers
These new modules will help in supporting training with speaker information:
* `CT <layer number>` - creates a ConcatTable from the given layer number and use the rest of the architecture file as the architecture for the new branch.
* `GRL <float number>` - creates GradientReverse layer with the given lambda value.
* `LSE` - creates LogSumExp over time.

## Training from scratch
Training acoustic model together with speaker information (multitask/adversarial learning).

### Arguments
* Provide the suffix of the speaker files using the `-speaker` argument, the default values for librispeech is  `spk`.
* Provide the path to the list of the speakers (the same as in the the letters/phones list), using the `-speakerslist` argument.
* Another parameter that one can play with is the `-lambdaloss`. This parameter is the scaling parameter between the two losses.

### Architecture
In order to add another branch to the network you need to add the desire architecture of the new branch AFTER! the regular architecture.
The last layer can contains the `NSPEAERS` token, and then the model automatically adds the number of speakers given by the speakers list.

For example:
```
regular arch file...
...
CT 9
GRL 0.00001
C 220 484 15 1
GLU
DO 0.5
C 242 NSPEAERS 1 1
LSE
```

## Using pretrained model
If you wish to use pretrained model as starting point you can do it while using the `--continue` argument together with the following:

### Arguments
* Provide the suffix of the speaker files using the `-speaker` argument, the default values for librispeech is  `spk`.
* Provide the path to the list of the speakers (the same as in the the letters/phones list), using the `-speakerslist` argument.
* Set the `-spkbranch` argument to true.
* Specify the location of the new branch architecture with `-brancharch` argument.
* Another parameter that one can play with is the `-lambdaloss`. This parameter is the scaling parameter between the two losses.

### Architecture
The architecture file should now contains only the new branch parts.

For example:
```
CT 9
GRL 0.00001
C 220 484 15 1
GLU
DO 0.5
C 242 NSPEAERS 1 1
LSE
```
