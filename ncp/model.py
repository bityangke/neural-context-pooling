from keras.layers import (Activation, BatchNormalization, Conv1D, Dense,
                          Dropout, Flatten, Input, MaxPooling1D)
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K


def neural_context_encoder(arch_prm, input_shape):
    """Modular architecture to encode context feature

    Parameters
    ----------
    arch_prm : dict

    """
    first_layer = True
    combo_list, after_combo = arch_prm['combo_list'], arch_prm['after_combo']

    nce_md = Sequential()
    if arch_prm['type'] == 'conv':
        for i, combo in enumerate(combo_list):
            for j, c_unit in enumerate(combo):
                c_unit['kwargs']['name'] = 'conv{}_{}'.format(i, j)
                c_unit['kwargs']['border_mode'] = 'same'
                c_unit['kwargs']['init'] = 'he_uniform'
                if 'activation' not in c_unit['kwargs']:
                    c_unit['kwargs']['activation'] = 'relu'

                if first_layer:
                    c_unit['kwargs']['input_shape'] = input_shape
                    first_layer = False

                nce_md.add(Conv1D(*c_unit['args'], **c_unit['kwargs']))

                if 'batchnorm' in c_unit:
                    nce_md.add(BatchNormalization(**c_unit['batchnorm']))

                if 'activation' in c_unit:
                    nce_md.add(Activation(c_unit['activation']))

                if 'dropout' in c_unit:
                    nce_md.add(Dropout(c_unit['dropout']))

            if 'max-pool' in after_combo[i]:
                pooling_prm = after_combo[i]['max-pool']
                nce_md.add(MaxPooling1D(**pooling_prm))
            if 'flatten' in after_combo[i]:
                nce_md.add(Flatten())
    else:
        raise ValueError('Invalid type in arch_prm. Set it to "conv".')

    return nce_md


def neural_context_model(num_actions, receptive_field, arch_prm):
    """Instanciate Keras model

    Parameters
    ----------
    num_actions : int
    receptive_field : tuple

    """
    input_src = Input(shape=receptive_field, name='context_over_time')
    model = neural_context_encoder(arch_prm, receptive_field)
    encoded_input = model(input_src)
    output_prob = Dense(num_actions, activation='softmax',
                        name='output_prob')(encoded_input)
    output_offsets = Dense(2 * num_actions,
                           name='output_offsets')(encoded_input)

    model = Model(input=[input_src],
                  output=[output_prob, output_offsets])
    return model


def neural_context_shallow_model(num_actions, receptive_field):
    """Instanciate Keras model

    Parameters
    ----------
    num_actions : int
    receptive_field : tuple

    """
    input_src = Input(shape=receptive_field, name='context_over_time')
    input_flattened = Flatten()(input_src)
    output_prob = Dense(num_actions, W_regularizer=l2(1e-5),
                        name='output_prob')(input_flattened)
    output_offsets = Dense(2 * num_actions,
                           name='output_offsets')(input_flattened)

    model = Model(input=[input_src],
                  output=[output_prob, output_offsets])
    return model


def set_learning_rate(model, lr_start):
    """Set learning rate

    Parameters
    ----------
    model : keras.model
        Instance of keras model
    lr_start : float
        Initial learning rate value

    """
    K.set_value(model.optimizer.lr, lr_start)
