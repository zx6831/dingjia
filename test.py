from utils.vector_manager import VectorManager

if __name__ == "__main__":
    vm = VectorManager(model_name="moka-ai/m3e-base")
    vm.build_index("./docs/test.txt")

