def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'dense-grasp':
        from .dense_grasp import DenseGraspNet
        return DenseGraspNet
    elif network_name == 'dense-atten-grasp':
        from .dense_attention import DenseAttenGraspNet
        return DenseAttenGraspNet
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))
