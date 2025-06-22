This provides the full code for the test orchestrator, an improved robot controller, a sample test definition file, and an analysis script to visualize the results.

### Project Structure

Project files. 

```
CartPole/
├── worlds/
│   └── cartpole_world.wbt         # main world file
│
├── controllers/
│   ├── orchestrator/              # The main testing supervisor
│   │   └── orchestrator.py
│   │
│   └── cartpole_pid_instrumented/   # The improved robot controller
│       └── cartpole_pid_instrumented.py
│
├── scenarios/                     # Test definitions
│   └── test_suite.yaml
│
└── analysis/                      # Script to plot results
    └── plot_results.py
```

### Step 1: Configure the Webots World (`cartpole_world.wbt`)

You need to make two changes to your world file:

1.  Set the **Supervisor**: In the `WorldInfo` node, set `supervisor` to `"orchestrator"`.
2.  Set the **Robot's Controller**: In your `Robot` definition, change the controller field to `controller "cartpole_pid_instrumented"`.

Make sure your robot's main node and the pole's endpoint have `DEF` names as they do in the provided file (`DEF ROBOT Robot` and `DEF POLE_ENDPOINT`).

### Step 2: The Instrumented Robot Controller

This is a refined version of your PID controller. It is simplified because the `orchestrator` will handle all supervision, logging, and resetting. It also includes critical fixes like **integral anti-windup**.

**`controllers/cartpole_pid_instrumented/cartpole_pid_instrumented.py`**
```python
import math
from controller import Robot, Motor, PositionSensor

class CartpolePIDController:
    """A cleaned-up PID controller for the robot."""

    def __init__(self, robot: Robot, time_step: int):
        self.robot = robot
        self.time_step = time_step
        self.time_step_s = time_step / 1000.0

        # Constants
        self.MAX_MOTOR_SPEED = 25.0
        self.INTEGRAL_LIMIT = 5.0  # Anti-windup limit

        # PID Gains (Tuned to be more stable than the original aggressive values)
        self.P_GAIN = 40.0
        self.I_GAIN = 5.0
        self.D_GAIN = 0.5

        # Initialize devices
        self.pole_sensor = self.robot.getDevice("polePosSensor")
        self.pole_sensor.enable(self.time_step)

        self.wheels = []
        for name in ['wheel1', 'wheel2', 'wheel3', 'wheel4']:
            wheel = self.robot.getDevice(name)
            wheel.setPosition(float('inf'))
            wheel.setVelocity(0.0)
            self.wheels.append(wheel)

        # State variables
        self.integral_sum = 0.0
        self.previous_position = 0.0

    def set_robot_speed(self, motor_speed: float):
        """Sets the velocity for all four wheels."""
        # Clamp the motor speed to the maximum
        clamped_speed = max(-self.MAX_MOTOR_SPEED, min(self.MAX_MOTOR_SPEED, motor_speed))
        for wheel in self.wheels:
            wheel.setVelocity(clamped_speed)

    def run_step(self):
        """Executes one step of the PID control loop."""
        # Get sensor reading
        position = self.pole_sensor.getValue()

        # PID calculations
        # Proportional term
        error = position

        # Integral term with anti-windup
        self.integral_sum += error * self.time_step_s
        self.integral_sum = max(-self.INTEGRAL_LIMIT, min(self.INTEGRAL_LIMIT, self.integral_sum))

        # Derivative term
        differential = (position - self.previous_position) / self.time_step_s

        # PID formula
        motor_speed = (self.P_GAIN * error) + (self.I_GAIN * self.integral_sum) + (self.D_GAIN * differential)
        
        # Apply control
        self.set_robot_speed(motor_speed)

        # Update state for next iteration
        self.previous_position = position

if __name__ == "__main__":
    # The robot's main loop
    robot_instance = Robot()
    TIME_STEP = int(robot_instance.getBasicTimeStep())
    controller = CartpolePIDController(robot_instance, TIME_STEP)

    while robot_instance.step(TIME_STEP) != -1:
        controller.run_step()
```

### Step 3: The Test Scenario Definition File

This YAML file defines the test suite. It's human-readable and easy to modify without changing the code.

**`scenarios/test_suite.yaml`**
```yaml
suite_name: "Pole_Balancing_Stability_V1"

# Global settings for all scenarios in this suite
global_settings:
  duration_s: 10.0
  failure_pole_angle_rad: 1.3  # Corresponds to min/maxStop
  failure_cart_position_m: 4.0

# List of individual test scenarios to run
scenarios:
  - name: "Step_Response"
    type: "step_response"
    description: "Tests the response to an initial non-zero angle."
    # Parameter to sweep: Apply a short force to create an initial angle.
    initial_kick_force_N: [20, 40, 60]

  - name: "Impulse_Disturbance"
    type: "impulse_disturbance"
    description: "Applies a lateral force to the pole mid-simulation."
    # Parameters to sweep:
    impulse_force_N: [30, 60]
    impulse_time_s: [2.0, 4.0] # When to apply the impulse

  - name: "Payload_Variation"
    type: "step_response"
    description: "Tests step response with different pole-end masses."
    initial_kick_force_N: [40] # Use a single kick force
    parameter_overrides:
      - node_def: "POLE_ENDPOINT"
        field: "mass"
        values: [0.02, 0.05, 0.1] # Test with 20g, 50g, and 100g mass
```

### Step 4: The Test Orchestrator (Supervisor)

This is the core of the framework. It reads the YAML, configures and runs each test, and saves the data.

**`controllers/orchestrator/orchestrator.py`**
```python
import sys
import os
import yaml
import csv
import json
from datetime import datetime
from pathlib import Path
from itertools import product
from controller import Supervisor

class TestOrchestrator(Supervisor):
    """
    A Webots Supervisor to automatically run a suite of physics-based tests.
    """
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())
        self.timestep_s = self.timestep / 1000.0

        # Get references to key simulation nodes
        self.robot_node = self.getFromDef("ROBOT")
        self.pole_end_node = self.getFromDef("POLE_ENDPOINT")
        if not self.robot_node or not self.pole_end_node:
            print("ERROR: Could not find required DEF nodes 'ROBOT' or 'POLE_ENDPOINT'.")
            sys.exit(1)

        # Get fields for manipulation
        self.robot_translation_field = self.robot_node.getField("translation")
        self.pole_end_mass_field = self.pole_end_node.getProtoField("mass")

        self.pole_sensor = self.getDevice("polePosSensor")
        self.pole_sensor.enable(self.timestep)

    def run_test_suite(self, config_path):
        """Loads and runs all scenarios from a YAML config file."""
        print(f"Loading test suite from: {config_path}")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        suite_name = config.get("suite_name", "Test_Suite")
        results_root = Path(f"../../results/{suite_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        results_root.mkdir(parents=True, exist_ok=True)
        print(f"Saving results to: {results_root.resolve()}")

        summary_file_path = results_root / "summary.csv"
        summary_headers = ['run_id', 'scenario_name', 'status', 'settling_time_s', 'overshoot_rad', 'steady_state_error_rad']
        with open(summary_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(summary_headers)

        for scenario in config["scenarios"]:
            self.execute_scenario(scenario, config["global_settings"], results_root, summary_file_path)

        print("Test suite finished.")
        self.simulationQuit(0)

    def expand_parameters(self, scenario):
        """Generates a list of all parameter combinations for a scenario."""
        param_keys = [k for k, v in scenario.items() if isinstance(v, list)]
        param_values = [scenario[k] for k in param_keys]
        
        if not param_keys:
            return [{}] # Return a single run with no specific parameters
        
        runs = []
        for combo in product(*param_values):
            run_params = dict(zip(param_keys, combo))
            runs.append(run_params)
        return runs

    def execute_scenario(self, scenario, globals, results_root, summary_path):
        """Configures and runs a single scenario for all its parameter combinations."""
        print(f"\n--- Running Scenario: {scenario['name']} ---")

        # Handle parameter overrides separately
        override_params = scenario.pop("parameter_overrides", [])
        
        # Expand sweep parameters
        base_runs = self.expand_parameters(scenario)
        
        # Expand override parameters
        all_runs = []
        if override_params:
            for base_run in base_runs:
                override_keys = [p['field'] for p in override_params]
                override_values = [p['values'] for p in override_params]
                for combo in product(*override_values):
                    new_run = base_run.copy()
                    new_run.update(dict(zip(override_keys, combo)))
                    all_runs.append(new_run)
        else:
            all_runs = base_runs
            
        for i, run_params in enumerate(all_runs):
            run_id = f"{scenario['name']}_{i+1:03d}"
            print(f"  > Running Test: {run_id} with params: {run_params}")

            # 1. Reset and Configure Simulation
            self.simulationReset()
            self.robot_node.restartController()
            self.step(self.timestep * 5) # Wait for controller to initialize

            # Apply parameter overrides
            for key, value in run_params.items():
                if key == "mass":
                    self.pole_end_mass_field.setSFFloat(value)

            # 2. Run Simulation and Log Data
            timeseries_data, status = self.run_simulation_loop(scenario, run_params, globals)

            # 3. Save Results
            run_dir = results_root / run_id
            run_dir.mkdir()

            # Save timeseries
            with open(run_dir / "timeseries.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['time_s', 'pole_angle_rad', 'cart_position_m'])
                writer.writerows(timeseries_data)

            # Save metadata
            metadata = {
                "run_id": run_id,
                "scenario": scenario,
                "parameters": run_params,
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
            with open(run_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)
            
            # 4. Calculate KPIs and update summary
            kpis = self.calculate_kpis(timeseries_data, status)
            with open(summary_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([run_id, scenario['name'], status, kpis['settling_time_s'], kpis['overshoot_rad'], kpis['steady_state_error_rad']])


    def run_simulation_loop(self, scenario, params, globals):
        """The main simulation loop for a single test run."""
        max_duration_s = globals['duration_s']
        log = []
        status = "Completed"

        # --- Initial Kick for Step Response ---
        if scenario['type'] == 'step_response':
            kick_force = params.get('initial_kick_force_N', 20)
            self.pole_end_node.addForce([kick_force, 0, 0], False)
            self.step(self.timestep * 2) # Apply force for a short duration
            self.pole_end_node.addForce([0, 0, 0], False) # Remove force

        # --- Main Loop ---
        start_time = self.getTime()
        while self.step(self.timestep) != -1:
            current_time = self.getTime() - start_time
            if current_time > max_duration_s:
                break

            # --- Apply Impulse Disturbance ---
            if scenario['type'] == 'impulse_disturbance':
                impulse_time = params.get('impulse_time_s', 2.0)
                if abs(current_time - impulse_time) < self.timestep_s:
                    impulse_force = params.get('impulse_force_N', 50)
                    self.pole_end_node.addForce([impulse_force, 0, 0], False)
                elif abs(current_time - (impulse_time + 0.1)) < self.timestep_s:
                     self.pole_end_node.addForce([0, 0, 0], False) # Remove force after 100ms

            # Log data
            pole_angle = self.pole_sensor.getValue()
            cart_pos = self.robot_translation_field.getSFVec3f()[0]
            log.append([current_time, pole_angle, cart_pos])
            
            # Check failure conditions
            if abs(pole_angle) > globals['failure_pole_angle_rad']:
                status = "Failed - Pole Angle Limit"
                break
            if abs(cart_pos) > globals['failure_cart_position_m']:
                status = "Failed - Cart Position Limit"
                break
        
        return log, status

    def calculate_kpis(self, data, status):
        """Calculate Key Performance Indicators from timeseries data."""
        kpis = {'settling_time_s': None, 'overshoot_rad': None, 'steady_state_error_rad': None}
        if status != "Completed" or not data:
            return kpis
        
        times, angles, _ = zip(*data)
        
        # Overshoot (max absolute angle)
        kpis['overshoot_rad'] = max(abs(angle) for angle in angles)

        # Steady-state error (average of last 10% of data)
        last_10_percent_idx = int(len(angles) * 0.9)
        kpis['steady_state_error_rad'] = sum(angles[last_10_percent_idx:]) / len(angles[last_10_percent_idx:])
        
        # Settling Time (time to stay within +/- 5% of overshoot)
        settling_threshold = 0.05 * kpis['overshoot_rad']
        for i in range(len(angles) - 1, -1, -1):
            if abs(angles[i]) > settling_threshold:
                kpis['settling_time_s'] = times[i]
                break
        
        return kpis

if __name__ == "__main__":
    # Path to the scenario definition file, relative to this script
    scenario_file = Path(__file__).parent.parent.parent / "scenarios" / "test_suite.yaml"
    
    orchestrator = TestOrchestrator()
    orchestrator.run_test_suite(scenario_file)
```

### Step 5: Data Analysis and Visualization Script

This script runs *after* the simulation is complete. It reads the output directory and generates plots for comparison.

**`analysis/plot_results.py`**
```python
import argparse
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(results_dir: Path):
    """Loads and plots results from a test suite run."""
    if not results_dir.is_dir():
        print(f"Error: Directory not found at '{results_dir}'")
        return

    summary_path = results_dir / "summary.csv"
    if not summary_path.exists():
        print(f"Error: 'summary.csv' not found in '{results_dir}'")
        return

    # Print the summary DataFrame
    print("--- Test Suite Summary ---")
    summary_df = pd.read_csv(summary_path)
    print(summary_df.to_string())
    print("-" * 25)

    # Group runs by scenario name for plotting
    grouped = summary_df.groupby('scenario_name')

    for scenario_name, group in grouped:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f"Scenario: {scenario_name}", fontsize=16)

        for _, run in group.iterrows():
            run_id = run['run_id']
            run_dir = results_dir / run_id
            
            # Load metadata to create a meaningful label
            with open(run_dir / "metadata.json", 'r') as f:
                metadata = json.load(f)
            params = metadata['parameters']
            label = ", ".join([f"{k}={v}" for k, v in params.items()])
            if not label: label = "Baseline"
            
            # Load and plot timeseries data
            timeseries_df = pd.read_csv(run_dir / "timeseries.csv")
            ax1.plot(timeseries_df['time_s'], timeseries_df['pole_angle_rad'], label=label)
            ax2.plot(timeseries_df['time_s'], timeseries_df['cart_position_m'], label=label)

        ax1.set_title("Pole Angle vs. Time")
        ax1.set_ylabel("Angle (radians)")
        ax1.grid(True)
        ax1.legend()

        ax2.set_title("Cart Position vs. Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Position (m)")
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot results from the Webots test orchestrator.")
    parser.add_argument("results_dir", type=str, help="Path to the root results directory of a test suite run.")
    args = parser.parse_args()
    
    plot_results(Path(args.results_dir))
```

### How to Run the Framework

1.  **Run the Simulation:** Open a terminal in the root of your project (`your_webots_project/`). Run Webots in batch mode. This will automatically start the `orchestrator`, which will run all tests and then exit.
    ```bash
    webots --batch --minimize worlds/cartpole_world.wbt
    ```
    You will see console output as the orchestrator progresses through the tests. A new `results/` directory will be created.

2.  **Analyze the Results:** Once the simulation is finished, run the analysis script, pointing it to the newly created results directory.
    ```bash
    # The results directory name will have a timestamp, so use the actual name
    python analysis/plot_results.py results/Pole_Balancing_Stability_V1_20231027_143000/
    ```
    This will first print the summary table of KPIs to your console and then display plots comparing the performance of each test run within a scenario.
