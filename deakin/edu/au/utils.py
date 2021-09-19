# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021.  Mohamed Reda Bouadjenek, Deakin University                       +
#               Email:  reda.bouadjenek@deakin.edu.au                                    +
#                                                                                        +
#       Licensed under the Apache License, Version 2.0 (the "License");                  +
#       you may not use this file except in compliance with the License.                 +
#       You may obtain a copy of the License at:                                         +
#                                                                                        +
#       http://www.apache.org/licenses/LICENSE-2.0                                       +
#                                                                                        +
#       Unless required by applicable law or agreed to in writing, software              +
#       distributed under the License is distributed on an "AS IS" BASIS,                +
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.         +
#       See the License for the specific language governing permissions and              +
#       limitations under the License.                                                   +
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import matplotlib.pyplot as plt


def learning_plots(history):
    plt.figure(figsize=(15, 4))
    ax1 = plt.subplot(1, 2, 1)
    for l in history.history:
        if l == 'loss' or l == 'val_loss':
            loss = history.history[l]
            plt.plot(range(1, len(loss) + 1), loss, label=l)

    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    ax2 = plt.subplot(1, 2, 2)
    for k in history.history:
        if 'accuracy' in k:
            loss = history.history[k]
            plt.plot(range(1, len(loss) + 1), loss, label=k)
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print('')
