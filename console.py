import os
import time
import threading
import sqlite3
import numpy as np
import tensorflow as tf
import zlib
import json
from queue import Queue
from cryptography.fernet import Fernet
from host_program import NeuralNetwork, train_neural_network, make_predictions

class Symbiont:
    def __init__(self, host_program):
        self.host_program = host_program
        self.input_shape = (10,)
        self.output_shape = 2
        self.model = self.initialize_or_load_model()
        self.data_queue = Queue()
        self.encryption_key = Fernet.generate_key()
        self.fernet = Fernet(self.encryption_key)
        self.db_path = "behavior_data.db"
        self.create_database()

    def initialize_or_load_model(self):
        model_dir = "saved_model"
        if os.path.exists(model_dir):
            model = tf.keras.models.load_model(model_dir)
            print("Model loaded successfully.")
        else:
            model = NeuralNetwork(self.input_shape, self.output_shape, 0.001)
            print("New model initialized.")
        return model

    def create_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS behavior_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data TEXT NOT NULL,
                label INTEGER NOT NULL
            )
        ''')
        conn.commit()
        conn.close()

    def store_data(self, data, label):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO behavior_data (data, label) VALUES (?, ?)
        ''', (json.dumps(data), label))
        conn.commit()
        conn.close()

    def fetch_data(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT data, label FROM behavior_data')
        rows = cursor.fetchall()
        conn.close()
        return rows

    def compress_data(self, data):
        return zlib.compress(data)

    def decompress_data(self, compressed_data):
        return zlib.decompress(compressed_data)

    def encrypt_data(self, data):
        return self.fernet.encrypt(data)

    def decrypt_data(self, encrypted_data):
        return self.fernet.decrypt(encrypted_data)

    def observe_behavior(self):
        while True:
            data_point = self.host_program.get_state()
            self.store_data(data_point['data'], data_point['label'])
            self.data_queue.put(data_point)
            time.sleep(1)

    def learn(self):
        while True:
            if not self.data_queue.empty():
                data_points = [self.data_queue.get() for _ in range(min(10, self.data_queue.qsize()))]
                X_train = np.array([dp['data'] for dp in data_points])
                y_train = np.array([dp['label'] for dp in data_points])
                self.model, _ = train_neural_network(self.model, X_train, y_train)
            time.sleep(5)

    def mutate(self):
        while True:
            if not self.data_queue.empty():
                last_data = self.data_queue.queue[-1]['data']
                last_data = np.expand_dims(last_data, axis=0)
                new_behavior = make_predictions(self.model, last_data)
                self.host_program.update_behavior(new_behavior)
            time.sleep(5)

    def self_encrypt(self):
        model_bytes = self.model.to_json().encode()
        compressed_model = self.compress_data(model_bytes)
        encrypted_model = self.encrypt_data(compressed_model)
        with open("encrypted_model.bin", "wb") as f:
            f.write(encrypted_model)

    def start(self):
        observer_thread = threading.Thread(target=self.observe_behavior)
        learner_thread = threading.Thread(target=self.learn)
        mutator_thread = threading.Thread(target=self.mutate)
        observer_thread.start()
        learner_thread.start()
        mutator_thread.start()

        while True:
            time.sleep(60)
            self.self_encrypt()

# Console Interface
def console_interface(symbiont):
    print("\nSymbiont Console Interface")
    while True:
        print("\nAvailable Commands:")
        print("1. Start Symbiont")
        print("2. Import External Data")
        print("3. View Database Entries")
        print("4. Encrypt Model")
        print("5. Exit")

        choice = input("Enter command number: ")

        if choice == "1":
            print("Starting Symbiont...")
            symbiont.start()

        elif choice == "2":
            file_path = input("Enter the path to the data file (CSV/JSON): ")
            if os.path.exists(file_path):
                try:
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                        data = df[df.columns[0]].to_list()
                        label = int(input("Enter the label for this data: "))
                        symbiont.store_data(data, label)
                        print("Data imported and stored successfully.")
                    elif file_path.endswith('.json'):
                        df = pd.read_json(file_path)
                        data = df[df.columns[0]].to_list()
                        label = int(input("Enter the label for this data: "))
                        symbiont.store_data(data, label)
                        print("Data imported and stored successfully.")
                    else:
                        print("Unsupported file format. Please use CSV or JSON.")
                except Exception as e:
                    print(f"Error importing data: {e}")
            else:
                print("File not found.")

        elif choice == "3":
            print("Database Entries:")
            rows = symbiont.fetch_data()
            for row in rows:
                print(f"Data: {row[0]}, Label: {row[1]}")

        elif choice == "4":
            print("Encrypting model...")
            symbiont.self_encrypt()
            print("Model encrypted successfully.")

        elif choice == "5":
            print("Exiting...")
            break

        else:
            print("Invalid command. Please try again.")

# Example usage:
if __name__ == "__main__":
    host_program = NeuralNetwork((10,), 2, 0.001)  # Replace with your actual host program initialization
    symbiont = Symbiont(host_program)
    console_interface(symbiont)