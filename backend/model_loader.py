# model manager
class ModelLoader:
    def __init__(self, path: str, name: str, version: int = 1.0, model_dir= ""):
        self.model_dir = model_dir
        self.name = name
        self.version = version
        self.path = path
        if self.model_dir == "sklearn_clf":
            self.model = self.__load_model_from_sklearn_clf(self.path)
        elif self.model_dir == "sklearn_svc":
            self.model = self.__load_model_from_sklearn_svc(self.path)
        elif self.model_dir == "sklearn_rfc":
            self.model = self.__load_model_from_sklearn_rfc(self.path)
        elif self.model_dir == "sklearn_m1":
            self.model = self.__load_model_from_sklearn_m1(self.path)
        else:
            raise NotImplementedError

    def __load_model_from_sklearn_clf(self, model_path):
        import pickle
        with open(model_path, "rb") as f:
            return pickle.load(f)
    def __load_model_from_sklearn_svc(self, model_path):
        import pickle
        with open(model_path, "rb") as f:
            return pickle.load(f)
    def __load_model_from_sklearn_rfc(self, model_path):
        import pickle
        with open(model_path, "rb") as f:
            return pickle.load(f)
    def __load_model_from_sklearn_m1(self, model_path):
        import pickle
        with open(model_path, "rb") as f:
            return pickle.load(f)

    def predict(self, data):
        return self.model.predict(data)
    
    def predict_pro(self, data):
        return self.model.predict_proba(data)

    def __call__(self, data):
        return self.predict(data)  # this is the same as the predict method
