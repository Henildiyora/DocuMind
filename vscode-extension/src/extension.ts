import * as vscode from 'vscode';
import * as path from 'path';
import * as child_process from 'child_process';

let pythonProcess: child_process.ChildProcess | undefined;
let outputChannel: vscode.OutputChannel;

let extensionContext: vscode.ExtensionContext;

export function activate(context: vscode.ExtensionContext) {
    extensionContext = context;
    outputChannel = vscode.window.createOutputChannel('DocuMind');
    outputChannel.appendLine('DocuMind extension activated');

    // Register commands
    context.subscriptions.push(
        vscode.commands.registerCommand('documind.startChat', startChat),
        vscode.commands.registerCommand('documind.ingestCodebase', ingestCodebase)
    );
}

export function deactivate() {
    if (pythonProcess) {
        pythonProcess.kill();
    }
    outputChannel.dispose();
}

async function startChat() {
    const config = vscode.workspace.getConfiguration('documind');
    const pythonPath = config.get<string>('pythonPath', 'python3');
    const mode = config.get<string>('mode', 'LOCAL');

    // Get workspace folder
    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (!workspaceFolder) {
        vscode.window.showErrorMessage('No workspace folder open');
        return;
    }

    // Check if main.py exists
    const mainPyPath = path.join(workspaceFolder.uri.fsPath, 'main.py');
    const fs = require('fs');
    if (!fs.existsSync(mainPyPath)) {
        vscode.window.showErrorMessage('main.py not found in workspace root. Please ensure DocuMind is properly set up.');
        return;
    }

    // Create webview panel for chat
    const panel = vscode.window.createWebviewPanel(
        'documindChat',
        'DocuMind Chat',
        vscode.ViewColumn.One,
        {
            enableScripts: true,
            retainContextWhenHidden: true
        }
    );

    // Set up webview HTML
    panel.webview.html = getWebviewContent();

    // Handle messages from webview
    panel.webview.onDidReceiveMessage(
        async (message) => {
            switch (message.type) {
                case 'sendMessage':
                    await handleChatMessage(message.text, panel, pythonPath, workspaceFolder.uri.fsPath, mode);
                    break;
                case 'ingest':
                    await runIngestion(pythonPath, workspaceFolder.uri.fsPath, mode);
                    break;
            }
        },
        undefined,
        extensionContext.subscriptions
    );

    outputChannel.appendLine('Chat interface opened');
}

async function ingestCodebase() {
    const config = vscode.workspace.getConfiguration('documind');
    const pythonPath = config.get<string>('pythonPath', 'python3');
    const mode = config.get<string>('mode', 'LOCAL');

    const workspaceFolder = vscode.workspace.workspaceFolders?.[0];
    if (!workspaceFolder) {
        vscode.window.showErrorMessage('No workspace folder open');
        return;
    }

    await runIngestion(pythonPath, workspaceFolder.uri.fsPath, mode);
}

async function runIngestion(pythonPath: string, workspacePath: string, mode: string) {
    return new Promise<void>((resolve, reject) => {
        const env = { ...process.env };

        // Set environment variables for the Python process
        if (mode === 'ONLINE') {
            const config = vscode.workspace.getConfiguration('documind');
            env.PINECONE_API_KEY = config.get<string>('pineconeApiKey', '');
            env.GOOGLE_API_KEY = config.get<string>('googleApiKey', '');
        }

        const pythonProcess = child_process.spawn(
            pythonPath,
            ['main.py', '--target', workspacePath, '--no-ingest'],
            {
                cwd: workspacePath,
                env: env,
                stdio: ['pipe', 'pipe', 'pipe']
            }
        );

        let output = '';
        pythonProcess.stdout?.on('data', (data) => {
            output += data.toString();
            outputChannel.append(data.toString());
        });

        pythonProcess.stderr?.on('data', (data) => {
            outputChannel.append(`Error: ${data.toString()}`);
        });

        pythonProcess.on('close', (code) => {
            if (code === 0) {
                vscode.window.showInformationMessage('Codebase ingestion completed successfully');
                resolve();
            } else {
                vscode.window.showErrorMessage(`Ingestion failed with code ${code}`);
                reject(new Error(`Process exited with code ${code}`));
            }
        });

        // For now, just run ingestion without interactive chat
        setTimeout(() => {
            pythonProcess.kill();
        }, 10000); // Kill after 10 seconds for testing
    });
}

async function handleChatMessage(
    message: string,
    panel: vscode.WebviewPanel,
    pythonPath: string,
    workspacePath: string,
    mode: string
) {
    // This is a simplified implementation
    // In a full implementation, you'd maintain a persistent Python process
    // and communicate with it via stdin/stdout or a socket

    panel.webview.postMessage({
        type: 'response',
        text: `Processing: ${message}\n\n(This is a placeholder. Full chat integration requires persistent Python process management.)`
    });
}

function getWebviewContent() {
    return `
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>DocuMind Chat</title>
            <style>
                body {
                    font-family: var(--vscode-font-family);
                    margin: 0;
                    padding: 20px;
                    background-color: var(--vscode-editor-background);
                    color: var(--vscode-editor-foreground);
                }
                #chat-container {
                    height: 400px;
                    overflow-y: auto;
                    border: 1px solid var(--vscode-panel-border);
                    padding: 10px;
                    margin-bottom: 10px;
                }
                #input-container {
                    display: flex;
                }
                #message-input {
                    flex: 1;
                    padding: 5px;
                    background-color: var(--vscode-input-background);
                    color: var(--vscode-input-foreground);
                    border: 1px solid var(--vscode-input-border);
                }
                #send-button {
                    margin-left: 5px;
                    padding: 5px 10px;
                    background-color: var(--vscode-button-background);
                    color: var(--vscode-button-foreground);
                    border: none;
                    cursor: pointer;
                }
                #ingest-button {
                    margin-top: 10px;
                    padding: 5px 10px;
                    background-color: var(--vscode-button-secondaryBackground);
                    color: var(--vscode-button-secondaryForeground);
                    border: none;
                    cursor: pointer;
                }
            </style>
        </head>
        <body>
            <h2>DocuMind AI Assistant</h2>
            <div id="chat-container"></div>
            <div id="input-container">
                <input type="text" id="message-input" placeholder="Ask DocuMind a question...">
                <button id="send-button">Send</button>
            </div>
            <button id="ingest-button">Ingest Codebase</button>

            <script>
                const vscode = acquireVsCodeApi();
                const chatContainer = document.getElementById('chat-container');
                const messageInput = document.getElementById('message-input');
                const sendButton = document.getElementById('send-button');
                const ingestButton = document.getElementById('ingest-button');

                sendButton.addEventListener('click', () => {
                    const message = messageInput.value.trim();
                    if (message) {
                        vscode.postMessage({ type: 'sendMessage', text: message });
                        messageInput.value = '';
                    }
                });

                ingestButton.addEventListener('click', () => {
                    vscode.postMessage({ type: 'ingest' });
                });

                messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        sendButton.click();
                    }
                });

                window.addEventListener('message', event => {
                    const message = event.data;
                    if (message.type === 'response') {
                        const responseDiv = document.createElement('div');
                        responseDiv.innerHTML = '<strong>DocuMind:</strong> ' + message.text.replace(/\\n/g, '<br>');
                        chatContainer.appendChild(responseDiv);
                        chatContainer.scrollTop = chatContainer.scrollHeight;
                    }
                });
            </script>
        </body>
        </html>
    `;
}