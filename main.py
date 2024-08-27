from symbiont import Symbiont
from host_program import NeuralNetwork

class HostProgram:
    def get_state(self):
        return {'data': [0.5] * 10, 'label': 1}

    def update_behavior(self, new_behavior):
        print("Comportamiento actualizado a:", new_behavior)

def main():
    host_program = HostProgram()
    symbiont = Symbiont(host_program)
    
    # Start the console interface after initializing the symbiont
    symbiont.start()
    symbiont.console_interface()

if __name__ == "__main__":
    main()