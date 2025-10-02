def print_joints(model):
    for j_id, j_name in enumerate(model.names):
        print(j_id, j_name, model.joints[j_id].shortname())


def print_frames(model):
    for i, frame in enumerate(model.frames):
        print(i, frame.name, frame.parent, frame.type)
