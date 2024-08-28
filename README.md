# Dueling DDQN

🚧👷🛑 Under Construction!!!

## Table of Contents

1. [Overview](#overview)
2. [Setup](#setup)
3. [Results](#results)
4. [Analysis](#analysis)

## Overview

This repository contains an implementation of Dueling Double Deep Q-Network (Dueling DDQN) learning using PyTorch.

## Setup

### Required Dependencies

It's recommended to use a Conda environment to manage dependencies and avoid conflicts. You can create and activate a new Conda environment with the following commands:

```bash
conda create -n rl python=3.11
conda activate rl
```

After activating the environment, install the required dependencies using:

```bash
pip install -r requirements.txt
```

### Running the Algorithm

You can run the algorithm on any supported Gymnasium Atari environment with a discrete action space using the following command:

```bash
python main.py --env 'MsPacmanNoFrameskip-v4'
```

#### Command-Line Arguments

- **Environment Selection**: Use `-e` or `--env` to specify the Gymnasium environment. The default is `None`, so you must specify an environment.
  
  Example:

  ```bash
  python main.py --env 'PongNoFrameskip-v4'
  ```

- **Number of Learning Steps**: Use `--n_steps` to define how many training steps the agent should undergo. The default is 1,000,000 steps.

  Example:

  ```bash
  python main.py --env 'BreakoutNoFrameskip-v4' --n_steps 200000
  ```

- **Parallel Environments**: Use `--n_envs` to specify the number of parallel environments to run during training. The default is 32 environments, optimizing the training process.

  Example:

  ```bash
  python main.py --env 'AsterixNoFrameskip-v4' --n_envs 16
  ```

- **Continue Training**: Use `--continue_training` to determine whether to continue training from saved weights. The default is `True`, allowing you to resume training from where you left off.

  Example:

  ```bash
  python main.py --env 'AsteroidsNoFrameskip-v4' --continue_training False
  ```

Using a Conda environment along with these flexible command-line options will help you efficiently manage your dependencies and customize the training process for your specific needs.

## Results

<table>
    <tr>
        <td>
            <p><b>AirRaid</b></p>
            <img src="environments/AirRaidNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Alien</b></p>
            <img src="environments/AlienNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Amidar</b></p>
            <img src="environments/AmidarNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/AirRaidNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AlienNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AmidarNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Assault</b></p>
            <img src="environments/AssaultNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Asterix</b></p>
            <img src="environments/AsterixNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Asteroids</b></p>
            <img src="environments/AsteroidsNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/AssaultNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AsterixNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/AsteroidsNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Atlantis</b></p>
            <img src="environments/AtlantisNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>BankHeist</b></p>
            <img src="environments/BankHeistNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>BattleZone</b></p>
            <img src="environments/BattleZoneNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/AtlantisNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BankHeistNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BattleZoneNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>BeamRider</b></p>
            <img src="environments/BeamRiderNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Berzerk</b></p>
            <img src="environments/BerzerkNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Bowling</b></p>
            <img src="environments/BowlingNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/BeamRiderNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BerzerkNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BowlingNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Boxing</b></p>
            <img src="environments/BoxingNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Breakout</b></p>
            <img src="environments/BreakoutNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Carnival</b></p>
            <img src="environments/CarnivalNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/BoxingNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/BreakoutNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/CarnivalNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Centipede</b></p>
            <img src="environments/CentipedeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>ChopperCommand</b></p>
            <img src="environments/ChopperCommandNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>CrazyClimber</b></p>
            <img src="environments/CrazyClimberNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/CentipedeNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/ChopperCommandNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/CrazyClimberNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<!--<table>
    <tr>
        <td>
            <p><b>Defender</b></p>
            <img src="environments/DefenderNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>DemonAttack</b></p>
            <img src="environments/DemonAttackNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/DefenderNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/DemonAttackNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>DoubleDunk</b></p>
            <img src="environments/DoubleDunkNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>ElevatorAction</b></p>
            <img src="environments/ElevatorActionNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Enduro</b></p>
            <img src="environments/EnduroNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/DoubleDunkNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/ElevatorActionNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/EnduroNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>FishingDerby</b></p>
            <img src="environments/FishingDerbyNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Freeway</b></p>
            <img src="environments/FreewayNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Frostbite</b></p>
            <img src="environments/FrostbiteNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/FishingDerbyNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/FreewayNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/FrostbiteNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Gopher</b></p>
            <img src="environments/GopherNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Gravitar</b></p>
            <img src="environments/GravitarNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Hero</b></p>
            <img src="environments/HeroNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/GopherNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/GravitarNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/HeroNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>IceHockey</b></p>
            <img src="environments/IceHockeyNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>JamesBond</b></p>
            <img src="environments/JamesbondNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>JourneyEscape</b></p>
            <img src="environments/JourneyEscapeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/IceHockeyNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/JamesbondNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/JourneyEscapeNoFrameskip-v4_running_avg_q.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Kangaroo</b></p>
            <img src="environments/KangarooNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Krull</b></p>
            <img src="environments/KrullNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>KungFuMaster</b></p>
            <img src="environments/KungFuMasterNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/KangarooNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/KrullNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/KungFuMasterNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>MontezumaRevenge</b></p>
            <img src="environments/MontezumaRevengeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>MsPacman</b></p>
            <img src="environments/MsPacmanNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>NameThisGame</b></p>
            <img src="environments/NameThisGameNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/MontezumaRevengeNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/MsPacmanNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/NameThisGameNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Phoenix</b></p>
            <img src="environments/PhoenixNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Pitfall</b></p>
            <img src="environments/PitfallNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Pong</b></p>
            <img src="environments/PongNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/PhoenixNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/PitfallNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/PongNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Pooyan</b></p>
            <img src="environments/PooyanNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>PrivateEye</b></p>
            <img src="environments/PrivateEyeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Qbert</b></p>
            <img src="environments/QbertNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/PooyanNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/PrivateEyeNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/QbertNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Riverraid</b></p>
            <img src="environments/RiverraidNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>RoadRunner</b></p>
            <img src="environments/RoadRunnerNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Robotank</b></p>
            <img src="environments/RobotankNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/RiverraidNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/RoadRunnerNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/RobotankNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Seaquest</b></p>
            <img src="environments/SeaquestNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Skiing</b></p>
            <img src="environments/SkiingNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Solaris</b></p>
            <img src="environments/SolarisNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/SeaquestNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/SkiingNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/SolarisNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>SpaceInvaders</b></p>
            <img src="environments/SpaceInvadersNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>StarGunner</b></p>
            <img src="environments/StarGunnerNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Tennis</b></p>
            <img src="environments/TennisNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/SpaceInvadersNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/StarGunnerNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/TennisNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>TimePilot</b></p>
            <img src="environments/TimePilotNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Tutankham</b></p>
            <img src="environments/TutankhamNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>UpNDown</b></p>
            <img src="environments/UpNDownNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/TimePilotNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/TutankhamNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/UpNDownNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>Venture</b></p>
            <img src="environments/VentureNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>VideoPinball</b></p>
            <img src="environments/VideoPinballNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>WizardOfWor</b></p>
            <img src="environments/WizardOfWorNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/VentureNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/VideoPinballNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/WizardOfWorNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>
<table>
    <tr>
        <td>
            <p><b>YarsRevenge</b></p>
            <img src="environments/YarsRevengeNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
        <td>
            <p><b>Zaxxon</b></p>
            <img src="environments/ZaxxonNoFrameskip-v4.gif" width="250" height="250"/>
        </td>
    </tr>
    <tr>
        <td>
            <img src="metrics/YarsRevengeNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
        <td>
            <img src="metrics/ZaxxonNoFrameskip-v4_metrics.png" width="250" height="250"/>
        </td>
    </tr>
</table>-->

## Analysis

## Acknowledgements

Special thanks to Phil Tabor, an excellent teacher! I highly recommend his [Youtube channel](https://www.youtube.com/machinelearningwithphil).
