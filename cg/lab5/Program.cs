using System;
using System.IO;
using System.Collections.Generic;
using OpenTK.Graphics.OpenGL;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;

class Program
{
    static void Main()
    {
        var gameSettings = GameWindowSettings.Default;
        var windowSettings = new NativeWindowSettings()
        {
            ClientSize = new Vector2i(800, 600),
            Title = "Refraction and Transparent Objects"
        };

        using (var game = new Game(gameSettings, windowSettings))
        {
            game.Run();
        }
    }
}

class Game : GameWindow
{
    private int _sphereVAO, _shaderProgram;
    private Vector3 _lightPos = new Vector3(2.0f, 4.0f, -2.0f);
    private float _refractiveIndex = 1.52f; // Default to glass

    public Game(GameWindowSettings gameSettings, NativeWindowSettings windowSettings)
        : base(gameSettings, windowSettings) { }

    protected override void OnLoad()
    {
        base.OnLoad();

        // Configure OpenGL
        GL.Enable(EnableCap.DepthTest);

        // Load shaders from the shaders/ folder
        _shaderProgram = LoadShaderProgram("shaders/vertex_shader.glsl", "shaders/fragment_shader.glsl");

        // Set up sphere VAO
        _sphereVAO = CreateSphereVAO();

        Console.WriteLine($"OpenGL Version: {GL.GetString(StringName.Version)}");
        Console.WriteLine($"GLSL Version: {GL.GetString(StringName.ShadingLanguageVersion)}");
    }

    protected override void OnRenderFrame(FrameEventArgs args)
    {
        base.OnRenderFrame(args);

        GL.ClearColor(0.2f, 0.2f, 0.2f, 1.0f); // Dark background
        GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

        GL.UseProgram(_shaderProgram);

        // Set projection matrix
        Matrix4 projection = Matrix4.CreatePerspectiveFieldOfView(MathHelper.DegreesToRadians(45.0f), Size.X / (float)Size.Y, 0.1f, 100.0f);
        GL.UniformMatrix4(GL.GetUniformLocation(_shaderProgram, "projection"), false, ref projection);

        // Set view matrix
        Matrix4 view = Matrix4.LookAt(new Vector3(0.0f, 3.0f, 10.0f), Vector3.Zero, Vector3.UnitY);
        GL.UniformMatrix4(GL.GetUniformLocation(_shaderProgram, "view"), false, ref view);

        // Set light and camera positions
        GL.Uniform3(GL.GetUniformLocation(_shaderProgram, "cameraPos"), 0.0f, 3.0f, 10.0f);
        GL.Uniform3(GL.GetUniformLocation(_shaderProgram, "lightPos"), _lightPos);
        GL.Uniform3(GL.GetUniformLocation(_shaderProgram, "lightColor"), 1.0f, 1.0f, 1.0f);

        // Set refractive index
        GL.Uniform1(GL.GetUniformLocation(_shaderProgram, "refractiveIndex"), _refractiveIndex);

        // Render transparent sphere
        GL.BindVertexArray(_sphereVAO);
        Matrix4 model = Matrix4.CreateTranslation(-1.5f, 0.0f, 0.0f);
        GL.UniformMatrix4(GL.GetUniformLocation(_shaderProgram, "model"), false, ref model);
        GL.Uniform3(GL.GetUniformLocation(_shaderProgram, "objectColor"), 0.5f, 0.8f, 1.0f); // Transparent blue
        GL.DrawElements(PrimitiveType.Triangles, 7200, DrawElementsType.UnsignedInt, 0);

        // Render opaque sphere
        model = Matrix4.CreateTranslation(1.5f, 0.0f, 0.0f);
        GL.UniformMatrix4(GL.GetUniformLocation(_shaderProgram, "model"), false, ref model);
        GL.Uniform3(GL.GetUniformLocation(_shaderProgram, "objectColor"), 1.0f, 0.3f, 0.3f); // Opaque red
        GL.DrawElements(PrimitiveType.Triangles, 7200, DrawElementsType.UnsignedInt, 0);

        SwapBuffers();
    }

    protected override void OnUpdateFrame(FrameEventArgs args)
    {
        base.OnUpdateFrame(args);

        if (KeyboardState.IsKeyDown(OpenTK.Windowing.GraphicsLibraryFramework.Keys.Up))
        {
            _refractiveIndex = Math.Clamp(_refractiveIndex + 0.01f, 1.0f, 2.5f);
            Console.WriteLine($"Refractive Index: {_refractiveIndex}");
        }
        else if (KeyboardState.IsKeyDown(OpenTK.Windowing.GraphicsLibraryFramework.Keys.Down))
        {
            _refractiveIndex = Math.Clamp(_refractiveIndex - 0.01f, 1.0f, 2.5f);
            Console.WriteLine($"Refractive Index: {_refractiveIndex}");
        }

        float lightRotationSpeed = 2.0f;
        if (KeyboardState.IsKeyDown(OpenTK.Windowing.GraphicsLibraryFramework.Keys.Left))
        {
            var rotation = Quaternion.FromAxisAngle(Vector3.UnitY, MathHelper.DegreesToRadians(-lightRotationSpeed));
            _lightPos = Vector3.Transform(_lightPos, rotation);
        }
        if (KeyboardState.IsKeyDown(OpenTK.Windowing.GraphicsLibraryFramework.Keys.Right))
        {
            var rotation = Quaternion.FromAxisAngle(Vector3.UnitY, MathHelper.DegreesToRadians(lightRotationSpeed));
            _lightPos = Vector3.Transform(_lightPos, rotation);
        }
    }

    private int CreateSphereVAO()
    {
        const int latitudeBands = 30;
        const int longitudeBands = 30;
        const float radius = 1.0f;

        var vertices = new List<float>();
        var indices = new List<uint>();

        for (int lat = 0; lat <= latitudeBands; lat++)
        {
            var theta = lat * MathF.PI / latitudeBands;
            var sinTheta = MathF.Sin(theta);
            var cosTheta = MathF.Cos(theta);

            for (int lon = 0; lon <= longitudeBands; lon++)
            {
                var phi = lon * 2 * MathF.PI / longitudeBands;
                var sinPhi = MathF.Sin(phi);
                var cosPhi = MathF.Cos(phi);

                var x = cosPhi * sinTheta;
                var y = cosTheta;
                var z = sinPhi * sinTheta;

                vertices.AddRange(new[] { x * radius, y * radius, z * radius, x, y, z });
            }
        }

        for (int lat = 0; lat < latitudeBands; lat++)
        {
            for (int lon = 0; lon < longitudeBands; lon++)
            {
                uint first = (uint)(lat * (longitudeBands + 1) + lon);
                uint second = first + (uint)longitudeBands + 1;

                indices.Add(first);
                indices.Add(second);
                indices.Add(first + 1);

                indices.Add(second);
                indices.Add(second + 1);
                indices.Add(first + 1);
            }
        }

        int vao = GL.GenVertexArray();
        GL.BindVertexArray(vao);

        int vbo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ArrayBuffer, vbo);
        GL.BufferData(BufferTarget.ArrayBuffer, vertices.Count * sizeof(float), vertices.ToArray(), BufferUsageHint.StaticDraw);

        int ebo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ElementArrayBuffer, ebo);
        GL.BufferData(BufferTarget.ElementArrayBuffer, indices.Count * sizeof(uint), indices.ToArray(), BufferUsageHint.StaticDraw);

        int positionLocation = GL.GetAttribLocation(_shaderProgram, "aPosition");
        GL.VertexAttribPointer(positionLocation, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
        GL.EnableVertexAttribArray(positionLocation);

        int normalLocation = GL.GetAttribLocation(_shaderProgram, "aNormal");
        GL.VertexAttribPointer(normalLocation, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
        GL.EnableVertexAttribArray(normalLocation);

        GL.BindVertexArray(0);

        return vao;
    }

    private int LoadShaderProgram(string vertexPath, string fragmentPath)
    {
        string vertexCode = File.ReadAllText(vertexPath);
        string fragmentCode = File.ReadAllText(fragmentPath);

        int vertexShader = GL.CreateShader(ShaderType.VertexShader);
        GL.ShaderSource(vertexShader, vertexCode);
        GL.CompileShader(vertexShader);
        CheckShaderCompileStatus(vertexShader, "Vertex");

        int fragmentShader = GL.CreateShader(ShaderType.FragmentShader);
        GL.ShaderSource(fragmentShader, fragmentCode);
        GL.CompileShader(fragmentShader);
        CheckShaderCompileStatus(fragmentShader, "Fragment");

        int program = GL.CreateProgram();
        GL.AttachShader(program, vertexShader);
        GL.AttachShader(program, fragmentShader);
        GL.LinkProgram(program);

        GL.DeleteShader(vertexShader);
        GL.DeleteShader(fragmentShader);

        return program;
    }

    private void CheckShaderCompileStatus(int shader, string shaderType)
    {
        GL.GetShader(shader, ShaderParameter.CompileStatus, out int status);
        if (status == 0)
        {
            string infoLog = GL.GetShaderInfoLog(shader);
            throw new Exception($"{shaderType} shader compilation failed: {infoLog}");
        }
    }

    protected override void OnUnload()
    {
        base.OnUnload();
        GL.DeleteVertexArray(_sphereVAO);
        GL.DeleteProgram(_shaderProgram);
    }
}
