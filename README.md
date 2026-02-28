# Nonogram Solver

Project made for the Image and Sound Processing course of the Graphics, Multimedia and Virtual Reality master program.

## Team members
- Popa Bogdan Gabriel
- Roșu Mihai Cosmin

## Details
Small program that takes an image with a nonogram and processes it to extract the nonogram information, then solves the obtained nonogram and prints the solution to the command prompt.

The program works by finding the biggest contour (box) inside the image, then splits this box in multiple smaller boxes and extracts the information in each box. With this, the nonogram can be recreated inside the program and can be solved. The solver itself is an enumerative backtracking solver.

To identify the digits written in the boxes, we used pytesseract as it provided the best results.
