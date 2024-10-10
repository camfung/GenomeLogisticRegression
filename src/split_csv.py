import csv
import sys


def save_first_n_rows(input_file, output_file, n):
    try:
        n = int(n)
        if n < 1:
            raise ValueError("Number of rows must be a positive integer.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    try:
        with open(input_file, mode="r", newline="", encoding="utf-8") as infile:
            reader = csv.reader(infile)

            with open(output_file, mode="w", newline="", encoding="utf-8") as outfile:
                writer = csv.writer(outfile)

                # Write header
                header = next(reader)
                writer.writerow(header)

                # Write first 'n' rows
                for i, row in enumerate(reader):
                    if i >= n:
                        break
                    writer.writerow(row)

        print(f"Successfully saved the first {n} rows to {output_file}.")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(
            "Usage: python script.py <input_csv_file> <output_csv_file> <number_of_rows>"
        )
    else:
        save_first_n_rows(sys.argv[1], sys.argv[2], sys.argv[3])
