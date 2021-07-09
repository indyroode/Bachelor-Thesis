"""Nengo implementations of STDP rules."""

import nengo
from nengo.builder import Builder
from nengo.builder.learning_rules import get_pre_ens, get_post_ens
from nengo.builder.operator import Operator
from nengo.builder.signal import Signal
from nengo.params import BoolParam, NumberParam, StringParam
import numpy as np
from nengo.builder.learning_rules import build_or_passthrough, get_post_ens, get_pre_ens
from nengo.synapses import SynapseParam, Lowpass
from nengo.params import Default
# ================
# Frontend objects
# ================
#
# These objects are the ones that you include in your model description.
# They are applied to specific connections between groups of neurons.


class STDP(nengo.learning_rules.LearningRuleType):
    """Spike-timing dependent plasticity rule."""

    # Used by other Nengo objects
    modifies = "weights"
    probeable = ("pre_trace", "post_trace", "pre_scale", "post_scale")

    # Parameters
    pre_tau = NumberParam("pre_tau", low=0, low_open=True)
    pre_amp = NumberParam("pre_amp", low=0, low_open=True)
    post_tau = NumberParam("post_tau", low=0, low_open=True)
    post_amp = NumberParam("post_amp", low=0, low_open=True)
    bounds = StringParam("bounds")
    max_weight = NumberParam("max_weight")
    min_weight = NumberParam("min_weight")

    def __init__(
            self,
            pre_tau=0.0168,
            post_tau=0.0337,
            pre_amp=1.0,
            post_amp=0.1,
            bounds="hard",
            max_weight=0.1,
            min_weight=-0.1,
            learning_rate=1e-9,
    ):
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.pre_amp = pre_amp
        self.post_amp = post_amp
        self.bounds = bounds
        self.max_weight = max_weight
        self.min_weight = min_weight
        super(STDP, self).__init__(learning_rate)


class TripletSTDP(nengo.learning_rules.LearningRuleType):
    """Triplet spike-timing dependent plasticity rule.

    From "Triplets of Spikes in a Model of Spike Timing-Dependent Plasticity",
    Pfister & Gerstner, 2006.
    Here we implement the full model.
    """

    # Used by other Nengo objects
    modifies = "weights"
    probeable = ("pre_trace1", "pre_trace2", "post_trace1", "post_trace2")

    # Parameters
    pre_tau = NumberParam("pre_tau", low=0, low_open=True)
    pre_taux = NumberParam("pre_taux", low=0, low_open=True)
    pre_amp2 = NumberParam("pre_amp2", low=0, low_open=True)
    pre_amp3 = NumberParam("pre_amp3", low=0, low_open=True)
    post_tau = NumberParam("post_tau", low=0, low_open=True)
    post_tauy = NumberParam("post_tauy", low=0, low_open=True)
    post_amp2 = NumberParam("post_amp2", low=0, low_open=True)
    post_amp3 = NumberParam("post_amp3", low=0, low_open=True)
    post_synapse = SynapseParam('post_synapse', default=None, readonly=True)
    pre_synapse = SynapseParam("pre_synapse", default=Lowpass(tau=0.005), readonly=True)
    nearest_spike = BoolParam("nearest_spike")

    def __init__(self, param_set="default", post_synapse=Default, pre_synapse=Default, nearest_spike=False, learning_rate=1e-9):
        """Uses parameter sets defined by Pfister & Gerstner, 2006."""
        self.pre_tau = 0.0168
        self.post_tau = 0.0337
        self.nearest_spike = nearest_spike
        self.pre_synapse = pre_synapse
        self.post_synapse = (
            self.pre_synapse if post_synapse is Default else post_synapse
        )
        if param_set == "default":
            self.pre_taux = 0.101
            self.post_tauy = 0.125
            self.pre_amp2 = 5e-10
            self.pre_amp3 = 6.2e-3
            self.post_amp2 = 7e-3
            self.post_amp3 = 2.3e-4
        elif param_set == "visual" and nearest_spike:
            self.pre_taux = 0.714
            self.post_tauy = 0.04
            self.pre_amp2 = 8.8e-11
            self.pre_amp3 = 5.3e-2
            self.post_amp2 = 6.6e-3
            self.post_amp3 = 3.1e-3
        elif param_set == "visual" and not nearest_spike:
            self.pre_taux = 0.101
            self.post_tauy = 0.125
            self.pre_amp2 = 5e-10
            self.pre_amp3 = 6.2e-3
            self.post_amp2 = 7e-3
            self.post_amp3 = 2.3e-4
        elif param_set == "hippocampal" and self.nearest_spike:
            self.pre_taux = 0.575
            self.post_tauy = 0.047
            self.pre_amp2 = 4.6e-3
            self.pre_amp3 = 9.1e-3
            self.post_amp2 = 3e-3
            self.post_amp3 = 7.5e-9
        elif param_set == "hippocampal" and not self.nearest_spike:
            self.pre_taux = 0.946
            self.post_tauy = 0.027
            self.pre_amp2 = 6.1e-3
            self.pre_amp3 = 6.7e-3
            self.post_amp2 = 1.6e-3
            self.post_amp3 = 1.4e-3
        else:
            raise ValueError("Only 'visual' and 'hippocampal' recognized.")

        super(TripletSTDP, self).__init__(learning_rate)


# ===============
# Backend objects
# ===============
#
# These objects let the Nengo core backend know how to implement the rules
# defined in the model through frontend objects. They require some knowledge
# of the low-level details of how the Nengo core backends works, and will
# be different depending on the backend on which the learning rule is implemented.
# The general architecture of the Nengo core backend is described at
#   https://www.nengo.ai/nengo/backend_api.html
# but in the context of learning rules, each learning rule needs a build function
# that is associated with a frontend object (through the `Builder.register`
# function) that sets up the signals and operators that implement the rule.
# Nengo comes with many general purpose operators that could be combined
# to implement a learning rule, but in most cases it is easier to implement
# them using a custom operator that does the delta update equation.
# See, for example, `step_stdp` in the `SimSTDP` operator to see where the
# learning rule's equation is actually specified. The build function exists
# mainly to make sure to all of the signals used in the operator are the
# correct ones. This requires some knowledge of the Nengo core backend,
# but for learning rules they are all very similar, and this could be made
# more convenient through some new abstractions; see
#  https://github.com/nengo/nengo/pull/553
#  https://github.com/nengo/nengo/pull/1149
# for some initial attempts at making this more convenient.


@Builder.register(STDP)
def build_stdp(model, stdp, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_trace = Signal(np.zeros(pre_activities.size), name="pre_trace")
    post_trace = Signal(np.zeros(post_activities.size), name="post_trace")
    pre_scale = Signal(np.zeros(model.sig[conn]["weights"].shape), name="pre_scale")
    post_scale = Signal(np.zeros(model.sig[conn]["weights"].shape), name="post_scale")

    model.add_op(
        SimSTDP(
            pre_activities,
            post_activities,
            pre_trace,
            post_trace,
            pre_scale,
            post_scale,
            model.sig[conn]["weights"],
            model.sig[rule]["delta"],
            learning_rate=stdp.learning_rate,
            pre_tau=stdp.pre_tau,
            post_tau=stdp.post_tau,
            pre_amp=stdp.pre_amp,
            post_amp=stdp.post_amp,
            bounds=stdp.bounds,
            max_weight=stdp.max_weight,
            min_weight=stdp.min_weight,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_trace"] = pre_trace
    model.sig[rule]["post_trace"] = post_trace
    model.sig[rule]["pre_scale"] = pre_scale
    model.sig[rule]["post_scale"] = post_scale

    model.params[rule] = None  # no build-time info to return


class SimSTDP(Operator):
    def __init__(
            self,
            pre_activities,
            post_activities,
            pre_trace,
            post_trace,
            pre_scale,
            post_scale,
            weights,
            delta,
            learning_rate,
            pre_tau,
            post_tau,
            pre_amp,
            post_amp,
            bounds,
            max_weight,
            min_weight,
    ):
        self.learning_rate = learning_rate
        self.pre_tau = pre_tau
        self.post_tau = post_tau
        self.pre_amp = pre_amp
        self.post_amp = post_amp
        self.bounds = str(bounds).lower()
        self.max_weight = max_weight
        self.min_weight = min_weight

        self.sets = []
        self.incs = []
        self.reads = [pre_activities, post_activities, weights]
        self.updates = [delta, pre_trace, post_trace, pre_scale, post_scale]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def post_activities(self):
        return self.reads[1]

    @property
    def post_scale(self):
        return self.updates[4]

    @property
    def post_trace(self):
        return self.updates[2]

    @property
    def pre_activities(self):
        return self.reads[0]

    @property
    def pre_scale(self):
        return self.updates[3]

    @property
    def pre_trace(self):
        return self.updates[1]

    @property
    def weights(self):
        return self.reads[2]

    def make_step(self, signals, dt, rng):
        pre_activities = signals[self.pre_activities]
        post_activities = signals[self.post_activities]
        pre_trace = signals[self.pre_trace]
        post_trace = signals[self.post_trace]
        pre_scale = signals[self.pre_scale]
        post_scale = signals[self.post_scale]
        weights = signals[self.weights]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        # Could be configurable
        pre_ampscale = 1.0
        post_ampscale = 1.0

        if self.bounds == "hard":

            def update_scales():
                pre_scale[...] = ((self.max_weight - weights) > 0.0).astype(
                    np.float64
                ) * pre_ampscale
                post_scale[...] = (
                        -((self.min_weight + weights) < 0.0).astype(np.float64)
                        * post_ampscale
                )

        elif self.bounds == "soft":

            def update_scales():
                pre_scale[...] = (self.max_weight - weights) * pre_ampscale
                post_scale[...] = (self.min_weight + weights) * post_ampscale

        elif self.bounds == "none":

            def update_scales():
                pre_scale[...] = pre_ampscale
                post_scale[...] = -post_ampscale

        else:
            raise RuntimeError(
                "Unsupported bounds type. Only 'hard', 'soft' and 'none' are supported."
            )

        def step_stdp():
            update_scales()
            pre_trace[...] += (dt / self.pre_tau) * (
                    -pre_trace + self.pre_amp * pre_activities
            )
            post_trace[...] += (dt / self.post_tau) * (
                    -post_trace + self.post_amp * post_activities
            )

            delta[...] = alpha * (
                    pre_scale * pre_trace[np.newaxis, :] * post_activities[:, np.newaxis]
                    + post_scale * post_trace[:, np.newaxis] * pre_activities -1000
            )

        return step_stdp


@Builder.register(TripletSTDP)
def build_tripletstdp(model, stdp, rule):
    conn = rule.connection
    pre_activities = model.sig[get_pre_ens(conn).neurons]["out"]
    post_activities = model.sig[get_post_ens(conn).neurons]["out"]
    pre_trace1 = Signal(np.zeros(pre_activities.size), name="pre_trace1")
    post_trace1 = Signal(np.zeros(post_activities.size), name="post_trace1")
    pre_trace2 = Signal(np.zeros(pre_activities.size), name="pre_trace2")
    post_trace2 = Signal(np.zeros(post_activities.size), name="post_trace2")
    scale = Signal(np.zeros(model.sig[conn]["weights"].shape), name="scale")
    scale_low = Signal(np.zeros(model.sig[conn]["weights"].shape), name="scale-low")
    post_filtered = build_or_passthrough(model, stdp.post_synapse, post_activities)

    model.add_op(
        SimTripletSTDP(
            post_filtered,
            pre_activities,
            post_activities,
            pre_trace1,
            post_trace1,
            pre_trace2,
            post_trace2,
            scale,
            scale_low,
            model.sig[conn]["weights"],
            model.sig[rule]["delta"],
            learning_rate=stdp.learning_rate,
            pre_tau=stdp.pre_tau,
            pre_taux=stdp.pre_taux,
            post_tau=stdp.post_tau,
            post_tauy=stdp.post_tauy,
            pre_amp2=stdp.pre_amp2,
            pre_amp3=stdp.pre_amp3,
            post_amp2=stdp.post_amp2,
            post_amp3=stdp.post_amp3,
            nearest_spike=stdp.nearest_spike,
        )
    )

    # expose these for probes
    model.sig[rule]["pre_trace1"] = pre_trace1
    model.sig[rule]["post_trace1"] = post_trace1
    model.sig[rule]["pre_trace2"] = pre_trace2
    model.sig[rule]["post_trace2"] = post_trace2

    model.params[rule] = None  # no build-time info to return


class SimTripletSTDP(Operator):
    def __init__(
            self,
            post_filtered,
            pre_activities,
            post_activities,
            pre_trace1,
            post_trace1,
            pre_trace2,
            post_trace2,
            scale,
            scale_low,
            weights,
            delta,
            learning_rate,
            pre_tau,
            pre_taux,
            post_tau,
            post_tauy,
            pre_amp2,
            pre_amp3,
            post_amp2,
            post_amp3,
            nearest_spike,
    ):
        self.learning_rate = learning_rate
        self.pre_tau = pre_tau
        self.pre_taux = pre_taux
        self.post_tau = post_tau
        self.post_tauy = post_tauy
        self.pre_amp2 = pre_amp2
        self.pre_amp3 = pre_amp3
        self.post_amp2 = post_amp2
        self.post_amp3 = post_amp3
        self.nearest_spike = nearest_spike

        self.sets = []
        self.incs = []
        self.reads = [pre_activities, post_activities, weights, scale, scale_low, post_filtered]
        self.updates = [delta, pre_trace1, post_trace1, pre_trace2, post_trace2]

    @property
    def delta(self):
        return self.updates[0]

    @property
    def post_activities(self):
        return self.reads[1]

    @property
    def post_trace1(self):
        return self.updates[2]

    @property
    def post_trace2(self):
        return self.updates[4]

    @property
    def pre_activities(self):
        return self.reads[0]

    @property
    def pre_trace1(self):
        return self.updates[1]

    @property
    def pre_trace2(self):
        return self.updates[3]

    @property
    def weights(self):
        return self.reads[2]

    @property
    def scale(self):
        return self.reads[3]

    @property
    def scale_low(self):
        return self.reads[4]

    @property
    def post_filtered(self):
        return self.reads[5]


    def make_step(self, signals, dt, rng):
        post_filtered = signals[self.post_filtered]
        pre_activities = signals[self.pre_activities]
        post_activities = signals[self.post_activities]
        pre_trace1 = signals[self.pre_trace1]
        post_trace1 = signals[self.post_trace1]
        pre_trace2 = signals[self.pre_trace2]
        post_trace2 = signals[self.post_trace2]
        scale = signals[self.scale]
        scale_low = signals[self.scale_low]
        weights = signals[self.weights]
        delta = signals[self.delta]
        alpha = self.learning_rate * dt

        pre_t1 = dt / self.pre_tau
        post_t1 = dt / self.post_tau
        pre_t2 = dt / self.pre_taux
        post_t2 = dt / self.post_tauy

        if self.nearest_spike:

            def update_beforedelta():
                # Trace 1 gets full update
                pre_trace1[...] += -pre_trace1 * pre_t1
                post_trace1[...] += -post_trace1 * post_t1
                pre_s = pre_activities > 0
                pre_trace1[pre_s] = pre_activities[pre_s] * pre_t1
                post_s = post_activities > 0
                post_trace1[post_s] = post_activities[post_s] * post_t1

                # Trace 2 gets spike update later
                pre_trace2[...] += -pre_trace2 * pre_t2
                post_trace2[...] += -post_trace2 * post_t2

            def update_afterdelta():
                pre_s = pre_activities > 0
                pre_trace2[pre_s] = pre_activities[pre_s] * pre_t2
                post_s = post_activities > 0
                post_trace2[post_s] = post_activities[post_s] * post_t2

            def check_weights():
                scale[...] = (0.3 - weights) * 1
                scale_low[...] = (weights > 0).astype(np.float64) * 1

        else:

            def update_beforedelta():
                # Trace 1 gets full update
                pre_trace1[...] += pre_t1 * (-pre_trace1 + pre_activities)
                post_trace1[...] += post_t1 * (-post_trace1 + post_activities)
                # Trace 2 gets spike update later
                pre_trace2[...] += pre_t2 * -pre_trace2
                post_trace2[...] += post_t2 * -post_trace2

            def update_afterdelta():
                pre_trace2[...] += pre_t2 * pre_activities
                post_trace2[...] += post_t2 * post_activities

        def step_tripletstdp():
            # Update first traces before weight update
            update_beforedelta()
            check_weights()
            post_squared = 0.01 * alpha * post_filtered * post_filtered
            forgetting = weights * post_squared[:, None]
            delta[...] = alpha * (
                    pre_trace1[np.newaxis, :]
                    * post_activities[:, np.newaxis]
                    * (self.pre_amp2 + self.pre_amp3 * post_trace2[:, np.newaxis])
                    - post_trace1[:, np.newaxis]
                    * pre_activities[np.newaxis, :]
                    * (self.post_amp2 + self.post_amp3 * pre_trace2[np.newaxis, :])
            )
            delta[...] += -forgetting
            # Update second traces after weight update§
            update_afterdelta()

        return step_tripletstdp
