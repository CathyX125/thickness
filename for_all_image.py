from oxide.oxid_thickness import fast_oxide_thickness as oxide_thickness
import sys
import glob, os
import csv

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python {} path".format(sys.argv[0]))
        exit(128)

    ruler_in_um = 50.0
    ruler_in_pixel = 256.0

    coef = ruler_in_um/ruler_in_pixel

    path_name = sys.argv[1]

    file_list = []
    os.chdir(path_name)
    for file_name in glob.glob("*.png"):
        file_list.append(file_name)

    with open('results.csv', 'w') as csvfile:
        fieldnames = ['File_name', 'Length', 'Black area', 'Grey area']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for file_name in file_list:
            print("File: ", file_name)

            length, black, grey = oxide_thickness(file_name)

            # print("length, black thickness, grey thickness in pixels: ", length, black/length, grey/length)
            length_um = coef * length
            black_um = coef * black/length
            grey_um = coef * grey/length

            print("length, black thickness, grey thickness in um: ", length_um, black_um, grey_um)

            writer.writerow({'File_name': file_name, 'Length': length_um, 'Black area': black_um, 'Grey area': grey_um})
