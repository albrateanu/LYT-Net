import tensorflow as tf
import math

class CosineDecayWithRestartsLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, min_lr, total_steps, first_decay_steps, t_mul=2.0, m_mul=1.0):
        super(CosineDecayWithRestartsLearningRateSchedule, self).__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.first_decay_steps = first_decay_steps
        self.t_mul = t_mul
        self.m_mul = m_mul
        self.alpha = min_lr / initial_lr

    def __call__(self, step):
        completed_fraction = step / self.total_steps
        i_restart = tf.floor(tf.math.log(1 - completed_fraction * (1 - self.t_mul)) / tf.math.log(self.t_mul))

        sum_r = (1 - self.t_mul ** i_restart) / (1 - self.t_mul)
        completed_fraction_since_restart = (completed_fraction - sum_r) / self.t_mul ** i_restart

        decay_steps = self.first_decay_steps * self.t_mul ** i_restart
        cosine_decay = 0.5 * (1 + tf.math.cos(math.pi * completed_fraction_since_restart))
        decayed = (1 - self.alpha) * cosine_decay + self.alpha
        new_lr = self.initial_lr * decayed

        return tf.where(step < self.total_steps, new_lr, self.min_lr)
