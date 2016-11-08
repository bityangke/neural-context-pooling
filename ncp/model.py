from keras.layers import Input, Conv1D, Dense, Flatten, MaxPooling1D
from keras.models import Model, Sequential
from keras import backend as K


def neural_context_encoder(arch_prm, input_shape):
    """Modular architecture to encode context feature

    Parameters
    ----------
    arch_prm : dict

    """
    first_layer = True
    init_shape = dict(input_shape=input_shape)
    combo_list, after_combo = arch_prm['combo_list'], arch_prm['after_combo']

    nce_md = Sequential()
    if arch_prm['type'] == 'conv':
        for i, combo in enumerate(combo_list):
            for j, c_unit in enumerate(combo):
                c_unit_name = 'conv{}_{}'.format(i, j)
                num_units, filter_size = 16, 3
                if 'num_units' in c_unit:
                    num_units = c_unit['num_units']
                if 'filter_size' in c_unit:
                    filter_size = c_unit['filter_size']

                nce_md.add(Conv1D(num_units, filter_size, name=c_unit_name,
                                  border_mode='same', activation='relu',
                                  **init_shape))
                if first_layer:
                    init_shape = {}
                    first_layer = False

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
