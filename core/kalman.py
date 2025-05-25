class KalmanFilter:
    def __init__(
        self, process_variance, measurement_variance, initial_estimate=0.0
    ):
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        self.estimate = initial_estimate
        self.error_estimate = 1.0

    def update_batch(self, measurements):
        estimates = []
        for measurement in measurements:
            self.error_estimate += self.process_variance
            kalman_gain = self.error_estimate / (
                self.error_estimate + self.measurement_variance
            )
            self.estimate += kalman_gain * (measurement - self.estimate)
            self.error_estimate *= 1 - kalman_gain
            estimates.append(self.estimate)
        return estimates
