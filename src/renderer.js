/**
 * WebGL2 renderer for fish simulation.
 * Exact visual parity with SDL2 renderer.c
 */

const FISH_LENGTH = 30.0;
const FISH_WIDTH = 12.0;
const TAIL_LENGTH = 15.0;
const TAIL_WIDTH = 8.0;
const FIN_LENGTH = 10.0;

// Colors (normalized to 0-1)
const COLOR_BG = [20/255, 30/255, 60/255, 1.0];
const COLOR_FISH = [255/255, 160/255, 50/255, 1.0];
const COLOR_OUTLINE = [200/255, 120/255, 30/255, 1.0];
const COLOR_FOOD = [100/255, 200/255, 100/255, 1.0];
const COLOR_WHITE = [1.0, 1.0, 1.0, 1.0];
const COLOR_BLACK = [0.0, 0.0, 0.0, 1.0];

class Renderer {
    constructor(canvas) {
        this.canvas = canvas;
        this.gl = canvas.getContext('webgl2');
        if (!this.gl) {
            throw new Error('WebGL2 not supported');
        }
        this._initShaders();
        this._initBuffers();
    }

    _initShaders() {
        const gl = this.gl;

        // Vertex shader - simple 2D with uniform color
        const vsSource = `#version 300 es
            in vec2 aPosition;
            uniform vec2 uResolution;
            void main() {
                // Convert pixel coords to clip space (-1 to 1)
                vec2 clipSpace = (aPosition / uResolution) * 2.0 - 1.0;
                gl_Position = vec4(clipSpace.x, -clipSpace.y, 0.0, 1.0);
            }
        `;

        // Fragment shader - solid color
        const fsSource = `#version 300 es
            precision mediump float;
            uniform vec4 uColor;
            out vec4 fragColor;
            void main() {
                fragColor = uColor;
            }
        `;

        const vs = this._compileShader(gl.VERTEX_SHADER, vsSource);
        const fs = this._compileShader(gl.FRAGMENT_SHADER, fsSource);

        this.program = gl.createProgram();
        gl.attachShader(this.program, vs);
        gl.attachShader(this.program, fs);
        gl.linkProgram(this.program);

        if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
            throw new Error('Shader link failed: ' + gl.getProgramInfoLog(this.program));
        }

        this.aPosition = gl.getAttribLocation(this.program, 'aPosition');
        this.uResolution = gl.getUniformLocation(this.program, 'uResolution');
        this.uColor = gl.getUniformLocation(this.program, 'uColor');
    }

    _compileShader(type, source) {
        const gl = this.gl;
        const shader = gl.createShader(type);
        gl.shaderSource(shader, source);
        gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
            throw new Error('Shader compile failed: ' + gl.getShaderInfoLog(shader));
        }
        return shader;
    }

    _initBuffers() {
        const gl = this.gl;
        this.vertexBuffer = gl.createBuffer();
        this.vao = gl.createVertexArray();
        gl.bindVertexArray(this.vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.enableVertexAttribArray(this.aPosition);
        gl.vertexAttribPointer(this.aPosition, 2, gl.FLOAT, false, 0, 0);
        gl.bindVertexArray(null);
    }

    resize(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
        this.gl.viewport(0, 0, width, height);
    }

    clear() {
        const gl = this.gl;
        gl.clearColor(COLOR_BG[0], COLOR_BG[1], COLOR_BG[2], COLOR_BG[3]);
        gl.clear(gl.COLOR_BUFFER_BIT);
    }

    _drawTriangles(vertices, color) {
        const gl = this.gl;
        gl.useProgram(this.program);
        gl.bindVertexArray(this.vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);
        gl.uniform2f(this.uResolution, this.canvas.width, this.canvas.height);
        gl.uniform4fv(this.uColor, color);
        gl.drawArrays(gl.TRIANGLES, 0, vertices.length / 2);
        gl.bindVertexArray(null);
    }

    _drawLines(vertices, color) {
        const gl = this.gl;
        gl.useProgram(this.program);
        gl.bindVertexArray(this.vao);
        gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(vertices), gl.DYNAMIC_DRAW);
        gl.uniform2f(this.uResolution, this.canvas.width, this.canvas.height);
        gl.uniform4fv(this.uColor, color);
        gl.drawArrays(gl.LINES, 0, vertices.length / 2);
        gl.bindVertexArray(null);
    }

    _quadToTriangles(x1, y1, x2, y2, x3, y3, x4, y4) {
        // Two triangles: (1,2,3) and (1,3,4)
        return [x1, y1, x2, y2, x3, y3, x1, y1, x3, y3, x4, y4];
    }

    _circleVertices(cx, cy, radius, segments = 16) {
        const verts = [];
        for (let i = 0; i < segments; i++) {
            const a1 = (i / segments) * Math.PI * 2;
            const a2 = ((i + 1) / segments) * Math.PI * 2;
            verts.push(cx, cy);
            verts.push(cx + Math.cos(a1) * radius, cy + Math.sin(a1) * radius);
            verts.push(cx + Math.cos(a2) * radius, cy + Math.sin(a2) * radius);
        }
        return verts;
    }

    drawFood(positions, count) {
        if (count === 0) return;
        const verts = [];
        for (let i = 0; i < count; i++) {
            const x = positions[i * 2];
            const y = positions[i * 2 + 1];
            verts.push(...this._circleVertices(x, y, 6, 12));
        }
        this._drawTriangles(verts, COLOR_FOOD);
    }

    drawFish(x, y, angle, tailPhase, tailAmplitude, bodyCurve, leftPec, rightPec) {
        const cos_a = Math.cos(angle);
        const sin_a = Math.sin(angle);

        // Body points (diamond shape) - matches renderer.c exactly
        const nose_x = x + cos_a * FISH_LENGTH;
        const nose_y = y + sin_a * FISH_LENGTH;

        const tail_base_x = x - cos_a * FISH_LENGTH * 0.7;
        const tail_base_y = y - sin_a * FISH_LENGTH * 0.7;

        const left_x = x - sin_a * FISH_WIDTH;
        const left_y = y + cos_a * FISH_WIDTH;

        const right_x = x + sin_a * FISH_WIDTH;
        const right_y = y - cos_a * FISH_WIDTH;

        // Draw body (filled quad: nose -> right -> tail -> left)
        const bodyVerts = this._quadToTriangles(
            nose_x, nose_y, right_x, right_y,
            tail_base_x, tail_base_y, left_x, left_y
        );
        this._drawTriangles(bodyVerts, COLOR_FISH);

        // Draw tail fin (animated by phase)
        const tail_swing = Math.sin(tailPhase) * 0.5 * tailAmplitude;
        const tail_angle = angle + Math.PI + tail_swing;
        const tail_cos = Math.cos(tail_angle);
        const tail_sin = Math.sin(tail_angle);

        const tail_tip_x = tail_base_x + tail_cos * TAIL_LENGTH;
        const tail_tip_y = tail_base_y + tail_sin * TAIL_LENGTH;
        const tail_left_x = tail_base_x - tail_sin * TAIL_WIDTH;
        const tail_left_y = tail_base_y + tail_cos * TAIL_WIDTH;
        const tail_right_x = tail_base_x + tail_sin * TAIL_WIDTH;
        const tail_right_y = tail_base_y - tail_cos * TAIL_WIDTH;

        const tailVerts = this._quadToTriangles(
            tail_base_x, tail_base_y, tail_left_x, tail_left_y,
            tail_tip_x, tail_tip_y, tail_right_x, tail_right_y
        );
        this._drawTriangles(tailVerts, COLOR_FISH);

        // Draw outline
        const outlineVerts = [
            nose_x, nose_y, right_x, right_y,
            right_x, right_y, tail_base_x, tail_base_y,
            tail_base_x, tail_base_y, left_x, left_y,
            left_x, left_y, nose_x, nose_y
        ];
        this._drawLines(outlineVerts, COLOR_OUTLINE);

        // Draw pectoral fins
        const pec_offset = 5.0;
        const finVerts = [];
        for (const side of [-1, 1]) {
            const pec_val = (side < 0) ? leftPec : rightPec;
            const pec_x = x - cos_a * pec_offset + sin_a * FISH_WIDTH * 0.6 * side;
            const pec_y = y - sin_a * pec_offset - cos_a * FISH_WIDTH * 0.6 * side;
            const fin_angle = angle + side * (1.5 - pec_val * 0.5);
            const fin_cos = Math.cos(fin_angle);
            const fin_sin = Math.sin(fin_angle);
            const fin_tip_x = pec_x + fin_cos * FIN_LENGTH;
            const fin_tip_y = pec_y + fin_sin * FIN_LENGTH;
            finVerts.push(pec_x, pec_y, fin_tip_x, fin_tip_y);
        }
        this._drawLines(finVerts, COLOR_OUTLINE);

        // Draw eye
        const eye_offset = 8;
        const eye_x = x + cos_a * eye_offset - sin_a * 4;
        const eye_y = y + sin_a * eye_offset + cos_a * 4;

        // White of eye (radius 3)
        this._drawTriangles(this._circleVertices(eye_x, eye_y, 3, 12), COLOR_WHITE);
        // Pupil (radius 1)
        this._drawTriangles(this._circleVertices(eye_x, eye_y, 1, 8), COLOR_BLACK);
    }
}
