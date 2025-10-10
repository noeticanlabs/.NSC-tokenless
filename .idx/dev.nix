{ pkgs, ... }:
let
  # Per the airules.md, we are defining a reproducible Nix-based environment.
  # This `let` block creates a custom Python environment containing all
  # necessary packages for the project. This is a more robust and declarative
  # approach than using pip.
  pythonWithPackages = pkgs.python3.withPackages (ps: with ps; [
    flask
    numpy
    scipy
    matplotlib
  ]);
in
{
  # The packages list makes the Python environment and LaTeX available in the workspace.
  packages = [
    pythonWithPackages
    pkgs.texlive.combined.scheme-small
  ];

  # The idx section configures the Firebase Studio workspace.
  idx = {
    # Install the recommended Python extension for VS Code.
    extensions = [
      "ms-python.python"
      "james-yu.latex-workshop"
    ];

    # Workspace lifecycle hooks, as described in airules.md.
    workspace = {
      # The onStart hook runs a command every time the workspace is started.
      onStart = {
        # This command starts the Flask server for the simulation.py app.
        # It uses the $PORT environment variable for dynamic port assignment.
        start-server = "${pythonWithPackages}/bin/python -m flask --app simulation.py run --host=0.0.0.0 --port=$PORT";
      };
    };

    # The previews section configures the web preview for the application.
    previews = {
      enable = true;
      previews = {
        web = {
          # The command to start the web preview. This must be a list of strings.
          command = [
            "${pythonWithPackages}/bin/python"
            "-m"
            "flask"
            "--app"
            "simulation.py"
            "run"
            "--host=0.0.0.0"
            "--port=$PORT"
          ];

          # The 'manager = "web"' property is required by the build system for web previews.
          manager = "web";
        };
      };
    };
  };
}
