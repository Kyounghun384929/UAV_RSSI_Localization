class KalmanFilter():
	def __init__(self, processNoise = 0.005, measurementNoise = 20):
		super(KalmanFilter, self).__init__()
		self.initialized = False
		self.processNoise = processNoise
		self.measurementNoise = measurementNoise
		self.predictedRSSI = 0
		self.errorCovariance = 0

	def filtering(self, rssi):
		if not self.isInitialized:
			self.isInitialized = True
			priorRSSI = rssi
			priorErrorCovariance = 1
		else:
			priorRSSI = self.predictedRSSI
			priorErrorCovariance = self.errorCovariance + self.processNoise

		kalmanGain = priorErrorCovariance / (priorErrorCovariance + self.measurementNoise)
		self.predictedRSSI = priorRSSI + (kalmanGain * (rssi - priorRSSI))
		self.errorCovarianceRSSI = (1 - kalmanGain) * priorErrorCovariance

		return self.predictedRSSI