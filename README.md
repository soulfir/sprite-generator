# Procedural Sprite Generator for 2D Game Characters

## Overview
This repository contains the implementation of a **Procedural Content Generation (PCG)** system for creating sprite sheets and animations for 2D humanoid game characters, as detailed in the paper:

> *Art and Animation: Procedural Content Generation for Sprite Sheet Creation*  
> Authors: Pedro M. Fernandes, Carlos Martinho, Rui Prada  
> Published in *Videogame Sciences and Arts (2025)*

The project leverages **cellular automata** to procedurally generate sprite sheets, offering diverse character designs and animations in a scalable, automated manner. This system is ideal for developers aiming to enhance variation and efficiency in 2D game development.

---

## Features

1. **Procedural Generation of Sprite Sheets:**
   - Supports front, back, left, and right views of characters.
   - Generates animations for basic movements in four directions.

2. **Cellular Automata Algorithm:**
   - Creates body parts with natural, organic shapes.
   - Ensures visual symmetry for humanoid appearance.

3. **Scalable Outputs:**
   - Supports different sprite sizes: 16x16, 32x32, and 64x64 pixels.
   - Optimized for small to medium-sized characters.

4. **Customizable GUI:**
   - A user-friendly interface to tweak parameters for art and animation generation.
   - Real-time sprite preview with adjustable animation speed.
   - Easy saving of generated sprites as PNGs and GIFs.

5. **High Diversity:**
   - Randomized parameters ensure diverse and unique character designs.
   - Custom color palette generation for consistent and harmonious sprite aesthetics.

---

## Applications
- **2D Games:** Automate sprite sheet creation for NPCs, enemies, and characters.
- **Game Prototyping:** Rapidly generate diverse assets for testing.
- **Artistic Exploration:** Experiment with unique designs and animations.

---

## Installation

### Requirements
- Python 3.8+
- Required Libraries:
  - `tkinter`
  - `Pillow`
  - `numpy`

### Installing `tkinter`
`tkinter` is a built-in library in Python for most distributions, but it may require additional installation depending on your system:

- **On Debian/Ubuntu-based systems:**
  ```bash
  sudo apt-get install python3-tk
  ```

- **On Fedora-based systems:**
  ```bash
  sudo dnf install python3-tkinter
  ```

- **On macOS:**
  `tkinter` is included with Python installations from python.org.

- **On Windows:**
  `tkinter` is included by default in Python installations.

For other libraries (`Pillow`, `numpy`), install them using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Clone Repository
```bash
git clone https://github.com/soulfir/sprite-generator.git
cd procedural-sprite-generator
```

---

## Usage

1. **Run the Sprite Generator GUI:**
   ```bash
   python sprite_generator_gui.py
   ```
2. **Features in GUI:**
   - Adjust sprite size and animation parameters using sliders.
   - Click `Generate!` to create a new sprite.
   - Preview animations in real time and save them as GIFs or sprite sheets.

3. **Direct Algorithm Execution (Optional):**
   For advanced users, run the core generator script:
   ```bash
   python npc_sprite_generator.py
   ```

---

## File Structure

- **`npc_sprite_generator.py`:** Core logic for procedural sprite generation using cellular automata.
- **`sprite_generator_gui.py`:** Graphical User Interface for customizing and generating sprites.
- **`assets/`:** Example generated sprites and animations.
- **`README.md`:** This documentation.
- **`requirements.txt`:** Python dependencies.

---

## Key Concepts

1. **Cellular Automata for Sprite Generation:**
   - Generates random matrices for organic shapes.
   - Applies symmetry for humanoid aesthetics.
   - Assigns colors using a custom color harmony algorithm.

2. **Animation Framework:**
   - Combines body components for sequential animations.
   - Ensures geometrical consistency across frames.

3. **GUI Features:**
   - Intuitive controls for size, layering, and animation speed.
   - Randomized sprite names for added character.

---

## Examples

Generated sprites and animations are located in the `assets/` directory. Examples include:
- **16x16:** Pixel-perfect miniatures ideal for retro-styled games.
- **32x32:** Balanced detail for modern 2D games.
- **64x64:** High-detail sprites suitable for larger characters or enemies.

---

## Future Improvements
1. Enhanced artistic control through palette and layer customization.
2. Automated filtering for low-quality or inappropriate sprites.
3. Group and species creation for themed sprite families.
4. Expanded animation capabilities for complex movements.

---

## Authors and Contributors

- **Pedro M. Fernandes**  
  INESC-ID and Instituto Superior Técnico, Lisbon, Portugal  
  Email: [pedro.miguel.rocha.fernandes@tecnico.ulisboa.pt](mailto:pedro.miguel.rocha.fernandes@tecnico.ulisboa.pt)

- **Carlos Martinho**  
  INESC-ID and Instituto Superior Técnico, Lisbon, Portugal  
  Email: [carlos.martinho@tecnico.ulisboa.pt](mailto:carlos.martinho@tecnico.ulisboa.pt)

- **Rui Prada**  
  INESC-ID and Instituto Superior Técnico, Lisbon, Portugal  
  Email: [rui.prada@tecnico.ulisboa.pt](mailto:rui.prada@tecnico.ulisboa.pt)

---

## Citation

If you use this work, please cite:
```bibtex
@inproceedings{procedural_sprite_generation_2025,
  title={Art and Animation: Procedural Content Generation for Sprite Sheet Creation},
  author={Fernandes, Pedro M. and Martinho, Carlos and Prada, Rui},
  booktitle={Videogame Sciences and Arts},
  year={2025},
  publisher={Springer}
}
```

---

## License

This project is licensed under the MIT License. See `LICENSE` for more details.
