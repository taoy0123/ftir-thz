from ftir.controller import Controller
from ftir.views import View  # adjust to your main window class


def main():
    view = View()
    controller = Controller(view)

    # Connect the "Load files" button to the controller.
    # Change load_button to the real attribute name in your view.
    view.load_button.configure(command=controller.load_files)
    view.save_button.configure(command=controller.save_csv)
    
    view.mainloop()


if __name__ == "__main__":
    main()