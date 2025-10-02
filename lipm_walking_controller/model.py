def print_joints(model):
    for j_id, j_name in enumerate(model.names):
        print(j_id, j_name, model.joints[j_id].shortname())


def print_frames(model):
    for i, frame in enumerate(model.frames):
        print(i, frame.name, frame.parent, frame.type)


def set_joint(q, model, joint_name, val):
    jid = model.getJointId(joint_name)
    if jid > 0 and model.joints[jid].nq == 1:
        q[model.joints[jid].idx_q] = val
