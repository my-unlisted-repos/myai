import torch

from ..datasets.mrislicer_utils import ReferenceSlice
from ..learner import CB, Learner


def get_learner(
    dltest,
    model,
    opt,
    loss,
    sched,
    reference_slice: ReferenceSlice,
    around: int,
    window_size = (96, 96),
    every: int | None = None,
    before_fit = False,
    after_fit = False,
    max_steps = 1000,
    render_video = True,
    fps = 30,
    postfix = None,
    extra_cbs = None,
) -> Learner:
    if extra_cbs is None: extra_cbs = ()
    if isinstance(extra_cbs, CB.Callback): extra_cbs = (extra_cbs,)

    ref_image, ref_seg = reference_slice.get((window_size), around)
    t1c = ref_image[0].clone()
    t1n = ref_image[1].clone()
    ref_image[0] = t1n
    ref_image[1] = t1c
    callbacks = [
        CB.StopOnStep(max_steps),
        CB.LogTime(),
        CB.Loss(),
        CB.Accelerate(),
        CB.Dice(['bg', 'necrosis', 'edema', 'tumor', 'resection'], train_step=8, bg_index=0),
        CB.PerformanceTweaks(True),
        CB.FastProgress(('train loss', 'test loss', 'train dice mean', 'test dice mean'), 0.2, 30, ybounds = (0, 1)),
        CB.Checkpoint(state_dict=False, in_epoch_dir=False).c_on('after_fit').c_on('on_fit_exception')
    ]
    if every is not None or before_fit or after_fit:
        callbacks.append(CB.test_epoch(dltest, every = every, before_fit = before_fit, after_fit = after_fit),)

    if render_video:
        callbacks.extend([
            CB.Renderer(fps=fps, nrows=4),
            CB.Render2DSegmentationVideo(
                inputs=ref_image.unsqueeze(0).to(torch.float32),
                targets=ref_seg.unsqueeze(0),
                n=1,
                nrows=2,
                activation=None,
                inputs_grid=True,
                overlay_channel=1,
                rgb_idxs=(1,2,3),
            ),
        ]
        )

    callbacks.extend(extra_cbs)

    learner = Learner(
        callbacks=callbacks,
        model=model,
        loss_fn=loss,
        optimizer=opt,
        scheduler=sched,
        main_metric = 'test dice mean',
    )

    if postfix is not None:
        learner.set_postfix(postfix)
    return learner
