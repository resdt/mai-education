using System;
using OpenTK.Graphics.OpenGL;
using OpenTK.Mathematics;
using OpenTK.Windowing.Common;
using OpenTK.Windowing.Desktop;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

class Program
{
    static void Main()
    {
        var gameSettings = GameWindowSettings.Default;
        var windowSettings = new NativeWindowSettings()
        {
            ClientSize = new Vector2i(800, 600),
            Title = "3D Textured Object"
        };

        using (var game = new Game(gameSettings, windowSettings))
        {
            game.Run();
        }
    }
}

class Game : GameWindow
{
    private int _texture1;
    private int _texture2;
    private int _shaderProgram;
    private int _vao;
    private int _vbo;
    private int _ebo;
    private bool _useTexture1 = true;
    private float _rotationAngle = 0.0f;
    private float _textureOffset = 0.0f;

    public Game(GameWindowSettings gameSettings, NativeWindowSettings windowSettings)
        : base(gameSettings, windowSettings) { }

    protected override void OnLoad()
    {
        base.OnLoad();

        GL.ClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        GL.Enable(EnableCap.DepthTest);

        // Load shaders
        _shaderProgram = CreateShaderProgram();
        GL.UseProgram(_shaderProgram);

        // Load textures
        _texture1 = LoadTexture("texture1.png");
        _texture2 = LoadTexture("texture2.png");

        // Define a 3D cube with texture coordinates
        float[] vertices = {
            // Positions          // Texture Coords
            -1.0f, -1.0f, -1.0f,  0.0f, 0.0f,
             1.0f, -1.0f, -1.0f,  1.0f, 0.0f,
             1.0f,  1.0f, -1.0f,  1.0f, 1.0f,
            -1.0f,  1.0f, -1.0f,  0.0f, 1.0f,
            -1.0f, -1.0f,  1.0f,  0.0f, 0.0f,
             1.0f, -1.0f,  1.0f,  1.0f, 0.0f,
             1.0f,  1.0f,  1.0f,  1.0f, 1.0f,
            -1.0f,  1.0f,  1.0f,  0.0f, 1.0f
        };

        uint[] indices = {
            0, 1, 2, 2, 3, 0, // Back face
            4, 5, 6, 6, 7, 4, // Front face
            0, 4, 7, 7, 3, 0, // Left face
            1, 5, 6, 6, 2, 1, // Right face
            3, 2, 6, 6, 7, 3, // Top face
            0, 1, 5, 5, 4, 0  // Bottom face
        };

        // Set up VAO, VBO, and EBO
        _vao = GL.GenVertexArray();
        GL.BindVertexArray(_vao);

        _vbo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ArrayBuffer, _vbo);
        GL.BufferData(BufferTarget.ArrayBuffer, vertices.Length * sizeof(float), vertices, BufferUsageHint.StaticDraw);

        _ebo = GL.GenBuffer();
        GL.BindBuffer(BufferTarget.ElementArrayBuffer, _ebo);
        GL.BufferData(BufferTarget.ElementArrayBuffer, indices.Length * sizeof(uint), indices, BufferUsageHint.StaticDraw);

        int positionLocation = GL.GetAttribLocation(_shaderProgram, "aPosition");
        GL.VertexAttribPointer(positionLocation, 3, VertexAttribPointerType.Float, false, 5 * sizeof(float), 0);
        GL.EnableVertexAttribArray(positionLocation);

        int texCoordLocation = GL.GetAttribLocation(_shaderProgram, "aTexCoord");
        GL.VertexAttribPointer(texCoordLocation, 2, VertexAttribPointerType.Float, false, 5 * sizeof(float), 3 * sizeof(float));
        GL.EnableVertexAttribArray(texCoordLocation);
    }

    protected override void OnRenderFrame(FrameEventArgs args)
    {
        base.OnRenderFrame(args);

        GL.Clear(ClearBufferMask.ColorBufferBit | ClearBufferMask.DepthBufferBit);

        GL.UseProgram(_shaderProgram);

        // Update rotation matrix
        Matrix4 model = Matrix4.CreateRotationY(MathHelper.DegreesToRadians(_rotationAngle));
        Matrix4 view = Matrix4.CreateTranslation(0.0f, 0.0f, -5.0f);
        Matrix4 projection = Matrix4.CreatePerspectiveFieldOfView(MathHelper.DegreesToRadians(45.0f), Size.X / (float)Size.Y, 0.1f, 100.0f);
        Matrix4 mvp = model * view * projection;

        int mvpLocation = GL.GetUniformLocation(_shaderProgram, "uMVP");
        GL.UniformMatrix4(mvpLocation, false, ref mvp);

        // Update texture offset
        int offsetLocation = GL.GetUniformLocation(_shaderProgram, "uTextureOffset");
        GL.Uniform1(offsetLocation, _textureOffset);

        // Bind texture
        GL.ActiveTexture(TextureUnit.Texture0);
        GL.BindTexture(TextureTarget.Texture2D, _useTexture1 ? _texture1 : _texture2);

        GL.BindVertexArray(_vao);
        GL.DrawElements(PrimitiveType.Triangles, 36, DrawElementsType.UnsignedInt, 0);

        SwapBuffers();
    }

    protected override void OnUpdateFrame(FrameEventArgs args)
    {
        base.OnUpdateFrame(args);

        _rotationAngle += 50.0f * (float)args.Time;
        _textureOffset += 0.5f * (float)args.Time;

        if (KeyboardState.IsKeyDown(OpenTK.Windowing.GraphicsLibraryFramework.Keys.Space))
        {
            _useTexture1 = !_useTexture1;
        }
    }

    protected override void OnUnload()
    {
        base.OnUnload();

        GL.DeleteTexture(_texture1);
        GL.DeleteTexture(_texture2);
        GL.DeleteProgram(_shaderProgram);
        GL.DeleteVertexArray(_vao);
        GL.DeleteBuffer(_vbo);
        GL.DeleteBuffer(_ebo);
    }

    private int CreateShaderProgram()
    {
        const string vertexShaderSource = @"#version 330 core
        layout (location = 0) in vec3 aPosition;
        layout (location = 1) in vec2 aTexCoord;

        uniform mat4 uMVP;
        uniform float uTextureOffset;

        out vec2 TexCoord;

        void main()
        {
            gl_Position = uMVP * vec4(aPosition, 1.0);
            TexCoord = aTexCoord + vec2(uTextureOffset, uTextureOffset);
        }";

        const string fragmentShaderSource = @"#version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;

        uniform sampler2D uTexture;

        void main()
        {
            FragColor = texture(uTexture, TexCoord);
        }";

        int vertexShader = CompileShader(ShaderType.VertexShader, vertexShaderSource);
        int fragmentShader = CompileShader(ShaderType.FragmentShader, fragmentShaderSource);

        int program = GL.CreateProgram();
        GL.AttachShader(program, vertexShader);
        GL.AttachShader(program, fragmentShader);
        GL.LinkProgram(program);

        GL.DeleteShader(vertexShader);
        GL.DeleteShader(fragmentShader);

        return program;
    }

    private int CompileShader(ShaderType type, string source)
    {
        int shader = GL.CreateShader(type);
        GL.ShaderSource(shader, source);
        GL.CompileShader(shader);

        GL.GetShader(shader, ShaderParameter.CompileStatus, out int status);
        if (status == 0)
        {
            string info = GL.GetShaderInfoLog(shader);
            throw new Exception($"Shader compilation failed: {info}");
        }

        return shader;
    }

    private int LoadTexture(string path)
    {
        using var image = Image.Load<Rgba32>(path);

        // Flip the image vertically
        var flippedImage = image.Clone(ctx => ctx.Flip(FlipMode.Vertical));

        // Convert the image to a byte array
        var pixelData = new byte[4 * flippedImage.Width * flippedImage.Height];
        flippedImage.CopyPixelDataTo(pixelData);

        int texture = GL.GenTexture();
        GL.BindTexture(TextureTarget.Texture2D, texture);

        GL.TexImage2D(TextureTarget.Texture2D, 0, PixelInternalFormat.Rgba,
            flippedImage.Width, flippedImage.Height, 0,
            OpenTK.Graphics.OpenGL.PixelFormat.Rgba, PixelType.UnsignedByte, pixelData);

        GL.GenerateMipmap(GenerateMipmapTarget.Texture2D);

        return texture;
    }
}
