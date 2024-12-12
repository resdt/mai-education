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
            Size = new Vector2i(800, 600),
            Title = "Soft Shadows with Directional Light",
            Flags = ContextFlags.ForwardCompatible,
            APIVersion = new Version(3, 3)
        };

        using (var game = new Game(gameSettings, windowSettings))
        {
            game.Run();
        }
    }
}

class Game : GameWindow
{
    private int _cubeVAO, _sphereVAO, _vbo, _ebo;
    private int _shaderProgram, _shadowShaderProgram;
    private int _depthMapFBO, _depthMap;

    private Vector3 _lightDir = new Vector3(-0.5f, -1.0f, -0.5f);
    private Matrix4 _lightSpaceMatrix;

    private bool _useSoftShadows = true;

    public Game(GameWindowSettings gameSettings, NativeWindowSettings windowSettings)
        : base(gameSettings, windowSettings) { }

    protected override void OnLoad()
    {
        base.OnLoad();

        GL.ClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        GL.Enable(EnableCap.DepthTest);

        // Load shaders
        _shaderProgram = LoadShaderProgram("shaders/vertex_shader.glsl", "shaders/fragment_shader.glsl");
        _shadowShaderProgram = LoadShaderProgram("shaders/shadow_vertex_shader.glsl", "shaders/shadow_fragment_shader.glsl");

        // Set up depth map framebuffer
        CreateDepthMap();

        // Create geometry
        _cubeVAO = CreateCube();
        _sphereVAO = CreateSphere(1.0f, 36, 18);

        // Calculate light space matrix
        Matrix4 lightProjection = Matrix4.CreateOrthographicOffCenter(-10, 10, -10, 10, 1.0f, 20.0f);
        Matrix4 lightView = Matrix4.LookAt(_lightDir * 10.0f, Vector3.Zero, Vector3.UnitY);
        _lightSpaceMatrix = lightProjection * lightView;

        Console.WriteLine("Press 'S' to toggle between soft and hard shadows.");
    }

    private void CreateDepthMap()
    {
        _depthMapFBO = GL.GenFramebuffer();
        _depthMap = GL.GenTexture();
        GL.BindTexture(TextureTarget.Texture2D, _depthMap);
        GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.DepthComponent, 1024, 1024, 0, PixelFormat.DepthComponent, PixelType.Float, IntPtr.Zero);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)TextureMinFilter.Nearest);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)TextureMagFilter.Nearest);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)TextureWrapMode.ClampToBorder);
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)TextureWrapMode.ClampToBorder);
        float[] borderColor = { 1.0f, 1.0f, 1.0f, 1.0f };
        GL.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBorderColor, borderColor);

        GL.BindFramebuffer(FramebufferTarget.Framebuffer, _depthMapFBO);
        GL.FramebufferTexture2D(FramebufferTarget.Framebuffer, FramebufferAttachment.DepthAttachment, TextureTarget.Texture2D, _depthMap, 0);
        GL.DrawBuffer(DrawBufferMode.None);
        GL.ReadBuffer(ReadBufferMode.None);

        // Check framebuffer completeness
        FramebufferErrorCode status = GL.CheckFramebufferStatus(FramebufferTarget.Framebuffer);
        if (status != FramebufferErrorCode.FramebufferComplete)
        {
            Console.WriteLine($"Framebuffer Error: {status}");
        }

        GL.BindFramebuffer(FramebufferTarget.Framebuffer, 0);
    }

    private int CreateCube()
    {
        float[] vertices = {
            // Positions          // Colors
            -0.5f, -0.5f, -0.5f,  1.0f, 0.0f, 0.0f,
             0.5f, -0.5f, -0.5f,  0.0f, 1.0f, 0.0f,
             0.5f,  0.5f, -0.5f,  0.0f, 0.0f, 1.0f,
            -0.5f,  0.5f, -0.5f,  1.0f, 1.0f, 0.0f,
            -0.5f, -0.5f,  0.5f,  1.0f, 0.0f, 1.0f,
             0.5f, -0.5f,  0.5f,  0.0f, 1.0f, 1.0f,
             0.5f,  0.5f,  0.5f,  1.0f, 1.0f, 1.0f,
            -0.5f,  0.5f,  0.5f,  0.0f, 0.0f, 0.0f,
        };

        uint[] indices = {
            0, 1, 2, 2, 3, 0, // Back face
            4, 5, 6, 6, 7, 4, // Front face
            0, 4, 7, 7, 3, 0, // Left face
            1, 5, 6, 6, 2, 1, // Right face
            3, 2, 6, 6, 7, 3, // Top face
            0, 1, 5, 5, 4, 0  // Bottom face
        };

        int vao = GL.GenVertexArray();
        GL.BindVertexArray(vao);

        _vbo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);
        GL.BufferData(BufferTarget.ArrayBuffer, vertices.Length * sizeof(float), vertices, BufferUsageHint.StaticDraw);

        _ebo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ElementArrayBuffer, _ebo);
        GL.BufferData(BufferTarget.ElementArrayBuffer, indices.Length * sizeof(uint), indices, BufferUsageHint.StaticDraw);

        GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
        GL.EnableVertexAttribArray(0);

        GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
        GL.EnableVertexAttribArray(1);

        GL.BindVertexArray(0);
        return vao;
    }

    private int CreateSphere(float radius, int sectors, int stacks)
    {
        var vertices = new List<float>();
        var indices = new List<uint>();

        for (int stack = 0; stack <= stacks; stack++)
        {
            float stackAngle = MathF.PI / 2 - stack * MathF.PI / stacks;
            float xy = radius * MathF.Cos(stackAngle);
            float z = radius * MathF.Sin(stackAngle);

            for (int sector = 0; sector <= sectors; sector++)
            {
                float sectorAngle = sector * 2 * MathF.PI / sectors;

                float x = xy * MathF.Cos(sectorAngle);
                float y = xy * MathF.Sin(sectorAngle);

                vertices.Add(x); vertices.Add(y); vertices.Add(z);
                vertices.Add(x / radius); vertices.Add(y / radius); vertices.Add(z / radius);
            }
        }

        for (int stack = 0; stack < stacks; stack++)
        {
            for (int sector = 0; sector < sectors; sector++)
            {
                int first = (stack * (sectors + 1)) + sector;
                int second = first + sectors + 1;

                indices.Add((uint)first);
                indices.Add((uint)second);
                indices.Add((uint)(first + 1));

                indices.Add((uint)second);
                indices.Add((uint)(second + 1));
                indices.Add((uint)(first + 1));
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

        GL.VertexAttribPointer(0, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 0);
        GL.EnableVertexAttribArray(0);

        GL.VertexAttribPointer(1, 3, VertexAttribPointerType.Float, false, 6 * sizeof(float), 3 * sizeof(float));
        GL.EnableVertexAttribArray(1);

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
        CheckProgramLinkStatus(program);

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

    private void CheckProgramLinkStatus(int program)
    {
        GL.GetProgram(program, GetProgramParameterName.LinkStatus, out int status);
        if (status == 0)
        {
            string infoLog = GL.GetProgramInfoLog(program);
            throw new Exception($"Shader linking failed: {infoLog}");
        }
    }

    protected override void OnRenderFrame(FrameEventArgs args)
    {
        base.OnRenderFrame(args);

        // Clear buffers
        GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

        // Use basic shader program
        GL.UseProgram(_shaderProgram);

        // Set projection and view matrices
        Matrix4 projection = Matrix4.CreatePerspectiveFieldOfView(MathHelper.DegreesToRadians(45.0f), Size.X / (float)Size.Y, 0.1f, 100.0f);
        Matrix4 view = Matrix4.LookAt(new Vector3(0.0f, 2.0f, 5.0f), Vector3.Zero, Vector3.UnitY);

        GL.UniformMatrix4(GL.GetUniformLocation(_shaderProgram, "projection"), false, ref projection);
        GL.UniformMatrix4(GL.GetUniformLocation(_shaderProgram, "view"), false, ref view);

        // Render cube
        GL.BindVertexArray(_cubeVAO);
        Matrix4 model = Matrix4.CreateTranslation(-1.5f, 0.0f, 0.0f);
        GL.UniformMatrix4(GL.GetUniformLocation(_shaderProgram, "model"), false, ref model);
        GL.DrawElements(PrimitiveType.Triangles, 36, DrawElementsType.UnsignedInt, 0);

        // Render sphere
        GL.BindVertexArray(_sphereVAO);
        model = Matrix4.CreateTranslation(1.5f, 0.0f, 0.0f);
        GL.UniformMatrix4(GL.GetUniformLocation(_shaderProgram, "model"), false, ref model);
        GL.DrawElements(PrimitiveType.Triangles, 7200, DrawElementsType.UnsignedInt, 0);

        // Swap buffers
        SwapBuffers();
    }


    private void RenderScene(int shader)
    {
        // Cube
        GL.BindVertexArray(_cubeVAO);
        Matrix4 model = Matrix4.CreateTranslation(-1.5f, 0.0f, 0.0f);
        GL.UniformMatrix4(GL.GetUniformLocation(shader, "model"), false, ref model);
        GL.DrawElements(PrimitiveType.Triangles, 36, DrawElementsType.UnsignedInt, 0);

        // Sphere
        GL.BindVertexArray(_sphereVAO);
        model = Matrix4.CreateTranslation(1.5f, 0.0f, 0.0f);
        GL.UniformMatrix4(GL.GetUniformLocation(shader, "model"), false, ref model);
        GL.DrawElements(PrimitiveType.Triangles, 7200, DrawElementsType.UnsignedInt, 0);
    }

    protected override void OnUpdateFrame(FrameEventArgs args)
    {
        base.OnUpdateFrame(args);

        if (KeyboardState.IsKeyDown(OpenTK.Windowing.GraphicsLibraryFramework.Keys.S))
        {
            _useSoftShadows = !_useSoftShadows;
            Console.WriteLine($"Soft Shadows: {_useSoftShadows}");
        }
    }

    protected override void OnUnload()
    {
        base.OnUnload();
        GL.DeleteFramebuffer(_depthMapFBO);
        GL.DeleteTexture(_depthMap);
        GL.DeleteProgram(_shaderProgram);
        GL.DeleteProgram(_shadowShaderProgram);
        GL.DeleteVertexArray(_cubeVAO);
        GL.DeleteVertexArray(_sphereVAO);
        GL.DeleteBuffer(_vbo);
        GL.DeleteBuffer(_ebo);
    }
}
