# Online Bayesian World Models, Fall 2026

This is a research-flavored project focused on developing and characterizing a (possibly) novel approach for efficiently adapating world models.

## Schedule and Resources

| **Week** | **Slides** | **Topics** | **Links** |
| --- | --- | --- | --- |
| 1 (9/27) | [Link](https://docs.google.com/presentation/d/11jPirH876BQy64SwzO0AYqlBr41xKZAsl3MtRCoDHVc/edit?usp=sharing) | Motivation, Foundations, Setup | --- |
| 2 (10/4) | [Link](https://docs.google.com/presentation/d/1WcVyPNqlrA2RvPb_LqLWh4jT0WLA7IZf61mfYfWomtY/edit?usp=sharing) | MPC, Regressions, and Adaptation | --- |
| 3 (10/11) | [Link](https://docs.google.com/presentation/d/1jeAcyVg2q30lpXUzTK76EUS8XmZMtOgwKyLqO6Gk_II/edit?usp=sharing) | Regressions, GPs, Updates | --- |

## Web Demo Instructions

Activate your virtual env first, then run:

```sh
python -m web
```

Select **Pusher → Load pusher demo → Play**. The rectangle must reach the blue
target pose with a 90° rotation. A rough patch resists both translation and
rotation. BLR and GP start empty and learn only from their own live transitions;
there are no calibration samples. The frozen model was trained on nominal physics
across all orientations. The planner is unchanged.

Reset clears adaptation and preserves physics. Restore settings returns to nominal
physics. Load pusher demo repeats the same setup. Dotted lines mark the patch center
± one width; its effect fades smoothly beyond them. **Rotation resistance** changes
how hard the rectangle is to turn independently of the patch.

See [the live-only comparison](results/pusher/online/RESULTS.md).

## Code Layout

```text
envs/                  drone, pendulum, and planar pusher; batched true dynamics
models/world_model.py  pretrained MLP and checkpoint loader
models/pretrained/     frozen model checkpoints
control/               CEM optimizer and MPC controllers
adaptation/            BLR, Gaussian processes, and residual composition
web/                   Flask server and canvas interface
experiments/pusher.py   matched prediction and MPC comparisons
results/pusher/        measurements, uncertainty plots, and demo
```

## AI Recommendations

It would be ridiculous and hypocritical for me to require manually coding all, or even the majority, of this project. However, it is important to note that delegating *all* implementation to an AI tool is known to have a clear negative impact on learning. Also, people have diverse opinions and habits regarding AI usage for code and research, so asserting any strict AI policy is probably a hopeless endeavor.

As a result, I will provide only *recommendations* for how you should use AI to contribute to this project:

1. Be comfortable with hand-deriving the math. There is (as of writing this) not a burdensome amount of math to know, and most of it is on the critical path for building a good intuition.

2. Understand at a conceptual level the requirements, function, and effects of each subsystem. Knowing exact module implementations and syntax is generally unnecessary, but being able to understand the purpose of large blocks of code is.

3. Treat your agents as pretty accurate explainers and decent writers, not full engineers. Most consumer agents are good at understanding the function of code and writing functional code. However, the decision-making quality for problems that do not already have exact or analogous answers already published is significantly worse than most competent technical people.

## Contact

Jeffrey Lu - `lujeff [at] umich [dot] edu`

## Acknowledgements

This project is supported by compute resources provided by [MIDAS](https://midas.umich.edu/), [U-M ARC High Performance Computing](https://its.umich.edu/advanced-research-computing).

This project is run through the [Michigan Data Science Team](https://mdst.club) in the Fall 2026 term.

## Gaussian Process Adapter

`GaussianProcessRegression` implements `partial_fit`, `predict`, and `reset`, like BLR.
Pass it to `AdaptiveWorldModel`; choose neural features or a `feature_fn` such as
`model.normalized_features`. The prior mean is zero, so an untouched adapter uses the frozen prediction.

The default GP keeps 256 recent observations. Set `max_samples=None` to retain all data.
`time_length=np.inf` models constant dynamics; a finite value adds exponential temporal decay.
Querying the model never advances its clock. The pusher demo uses the stationary GP;
changing settings resets both posteriors when the existing toggle is enabled.

The pusher uses a rectangular slider with guided contact, a quadratic limit surface,
and Coulomb sticking/sliding. It omits inertia, impacts, and contact loss; it is not PushT.
See [the live-only comparison](results/pusher/online/RESULTS.md).
