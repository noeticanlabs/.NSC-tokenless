{ pkgs, ... }:
{
  # The packages list makes the Node.js environment available in the workspace.
  packages = [
    pkgs.nodejs_20
  ];

  # The idx section configures the Firebase Studio workspace.
  idx = {
    # Install the recommended extensions for VS Code.
    extensions = [
      "dbaeumer.vscode-eslint"
      "esbenp.prettier-vscode"
      "csstools.postcss"
    ];

    # Workspace lifecycle hooks, as described in airules.md.
    workspace = {
      # The onCreate hook runs commands when the workspace is first created.
      onCreate = {
        npm-install = "npm install";
      };
      # The onStart hook runs a command every time the workspace is started.
      onStart = {
        # This command starts the Vite development server.
        start-server = "npm run dev";
      };
    };

    # The previews section configures the web preview for the application.
    previews = {
      enable = true;
      previews = {
        web = {
          # The command to start the web preview. This must be a list of strings.
          command = [ "npm" "run" "dev" "--" "--port" "$PORT" ];
          manager = "web";
        };
      };
    };
  };
}
