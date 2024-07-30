const std = @import("std");

const Code = @import("huffman_code.zig");

pub fn decode(alloc: std.mem.Allocator, reader: anytype) !struct {
    width: u16,
    height: u16,
    data: []Color,
} {
    var hdr: [21]u8 = undefined;
    if (21 != try reader.readAll(&hdr)) {
        return error.NotLosslessWebp;
    }

    if (!std.mem.eql(u8, hdr[0..4], "RIFF")) return error.NotLosslessWebp;
    const riff_len = std.mem.readInt(u32, hdr[4..8], .little);
    if (!std.mem.eql(u8, hdr[8..16], "WEBPVP8L")) return error.NotLosslessWebp;
    const stream_len = std.mem.readInt(u32, hdr[16..20], .little);
    if (hdr[20] != 0x2F) return error.NotLosslessWebp;
    // standard webp encoder is sometimes off-by-1 on stream_len :/
    // if (stream_len > riff_len -| 13) return error.Invalid;
    _ = riff_len;
    var lim_rd_state = std.io.limitedReader(reader, stream_len);
    var rd = std.io.bitReader(.little, lim_rd_state.reader());

    const sx = (try rd.readBitsNoEof(u16, 14)) + 1;
    const sy = (try rd.readBitsNoEof(u16, 14)) + 1;
    _ = try rd.readBitsNoEof(u8, 1); // alpha used hint
    if (0 != try rd.readBitsNoEof(u8, 3)) {
        return error.UnsupportedVersion;
    }

    const TfType = enum { pred, col, sub_green, palette };
    var transform_order = std.BoundedArray(TfType, 4){};

    var palette_tf: PaletteTf = undefined;
    var color_tf: ColorDecorellationTf = undefined;
    var pred_tf: PredictTf = undefined;

    var img = Image{ .data = try alloc.alloc(Color, @as(u32, sx) * sy), .sx = sx, .sy = sy };
    errdefer alloc.free(img.data);

    defer for (transform_order.slice()) |tf| switch (tf) {
        .col => alloc.free(color_tf.img.data),
        .pred => alloc.free(pred_tf.img.data),
        .sub_green, .palette => {},
    };

    while (1 == try rd.readBitsNoEof(u1, 1)) {
        const tf: TfType = @enumFromInt(try rd.readBitsNoEof(u2, 2));

        if (std.mem.indexOfScalar(TfType, transform_order.slice(), tf) != null)
            return error.Invalid;

        switch (tf) {
            .col => try color_tf.read(&rd, alloc, img),
            .pred => try pred_tf.read(&rd, alloc, img),
            .sub_green => {},
            .palette => try palette_tf.read(&rd, alloc, &img),
        }

        transform_order.append(tf) catch unreachable;
    }

    try decodeSubImage(alloc, true, &rd, img);

    for (0..transform_order.len) |i| switch (transform_order.buffer[transform_order.len - i - 1]) {
        .col => color_tf.invert(img),
        .pred => try pred_tf.invert(img),
        .sub_green => for (img.data) |*v| {
            v.r +%= v.g;
            v.b +%= v.g;
        },
        .palette => try palette_tf.invert(&img),
    };

    return .{
        .width = img.sx,
        .height = img.sy,
        .data = img.data,
    };
}

fn decodeSubImage(alloc: std.mem.Allocator, comptime meta_en: bool, rd: anytype, img: Image) !void {
    const sx = img.sx;
    const sy = img.sy;
    std.debug.assert(img.data.len == @as(u32, sx) * sy);

    const cache_en = 1 == try rd.readBitsNoEof(u1, 1);

    var cache_size: u16 = 0;
    var hash_shift: u5 = undefined;

    if (cache_en) {
        const cache_size_log = try rd.readBitsNoEof(u4, 4);
        if (cache_size_log == 0 or cache_size_log > 11) {
            return error.Invalid;
        }
        cache_size = @as(u16, 1) << cache_size_log;
        hash_shift = 31 - @as(u5, cache_size_log - 1);
    }

    var code_idx_img: ?[]Color = null;
    defer if (code_idx_img) |v| alloc.free(v);

    var block_size_log: u4 = 0;
    var code_img_stride: u16 = 0;
    var num_code_groups: u32 = 1;

    if (meta_en and 1 == try rd.readBitsNoEof(u1, 1)) {
        block_size_log = (try rd.readBitsNoEof(u4, 3)) + 2;
        const csx = divExpCeil(sx, block_size_log);
        const csy = divExpCeil(sy, block_size_log);

        code_img_stride = csx;
        code_idx_img = try alloc.alloc(Color, @as(u32, csx) * csy);
        try decodeSubImage(alloc, false, rd, .{ .data = code_idx_img.?, .sx = csx, .sy = csy });
        for (code_idx_img.?) |c| {
            num_code_groups = @max(num_code_groups, c.g + (@as(u16, c.r) << 8) + 1);
        }
    }

    const code_group_arr = try alloc.alloc(struct {
        main: Code,
        r: Code,
        b: Code,
        a: Code,
        dist: Code,
    }, num_code_groups);
    defer alloc.free(code_group_arr);

    const code_names = .{ "main", "r", "b", "a", "dist" };

    for (code_group_arr) |*cg| {
        inline for (code_names) |name| {
            // freeing an empty array is a noop
            @field(cg, name).map = &.{};
        }
    }

    defer for (code_group_arr) |*cg| {
        inline for (code_names) |name| {
            alloc.free(@field(cg, name).map);
        }
    };

    for (code_group_arr) |*cg| {
        var lens_buf: [256 + 24 + (1 << 11)]u8 = undefined;

        inline for (.{
            .{ "main", 256 + 24 + cache_size },
            .{ "r", 256 },
            .{ "b", 256 },
            .{ "a", 256 },
            .{ "dist", 40 },
        }) |tmp| {
            const name = tmp[0];
            const sym_count = tmp[1];
            @field(cg, name) = Code{ .map = try alloc.alloc(u16, sym_count) };
            try readCode(lens_buf[0..sym_count], rd);
            try @field(cg, name).build(lens_buf[0..sym_count]);
        }
    }

    var widx: u32 = 0;

    var color_cache: [1 << 11]Color = undefined;
    @memset(color_cache[0..cache_size], .{});

    var x: u16 = 0;
    var y: u16 = 0;

    const data = img.data;

    while (widx < data.len) {
        const code_group = if (code_idx_img) |i| blk: {
            const bx = x >> block_size_log;
            const by = y >> block_size_log;
            const col = i[by * code_img_stride + bx];
            break :blk code_group_arr[col.g + (@as(u16, col.r) << 8)];
        } else code_group_arr[0];

        switch (try code_group.main.readSym(rd)) {
            0...255 => |v| {
                const col = Color{
                    .r = @intCast(try code_group.r.readSym(rd)),
                    .g = @intCast(v),
                    .b = @intCast(try code_group.b.readSym(rd)),
                    .a = @intCast(try code_group.a.readSym(rd)),
                };

                if (cache_en) color_cache[hash(col, hash_shift)] = col;

                data[widx] = col;

                widx += 1;
                x += 1;
                if (x >= sx) {
                    x = 0;
                    y += 1;
                }
            },
            256...255 + 24 => |v| {
                const len = try readExtraBitsLZ(v - 256, rd);
                var dist = try readExtraBitsLZ(try code_group.dist.readSym(rd), rd);
                dist = mapDistance(dist, sx);

                if (len > data[widx..].len) return error.Invalid;
                if (dist > widx) return error.Invalid;

                for (data[widx - dist ..][0..len], data[widx..][0..len]) |src, *dst| {
                    dst.* = src;
                    if (cache_en) color_cache[hash(src, hash_shift)] = src;
                }

                widx += len;

                x = @intCast(widx % sx);
                y = @intCast(widx / sx);
            },
            else => |v| {
                data[widx] = color_cache[v - (256 + 24)];

                widx += 1;
                x += 1;
                if (x >= sx) {
                    x = 0;
                    y += 1;
                }
            },
        }
    }
}

const PaletteTf = struct {
    palette: [256]Color = undefined,
    sub_bits: u2 = 0,
    orig: Image = undefined,
    pub fn read(self: *PaletteTf, rd: anytype, alloc: std.mem.Allocator, img: *Image) !void {
        @memset(&self.palette, .{});
        self.sub_bits = 0;
        const pal_size = (try rd.readBitsNoEof(u16, 8)) + 1;

        try decodeSubImage(alloc, false, rd, .{ .data = self.palette[0..pal_size], .sx = pal_size, .sy = 1 });

        var acc = self.palette[0];
        for (self.palette[1..pal_size]) |*v| {
            acc = addColor(acc, v.*);
            v.* = acc;
        }

        if (pal_size <= 16) {
            self.sub_bits = 1;
        } else if (pal_size <= 4) {
            self.sub_bits = 2;
        } else if (pal_size <= 2) {
            self.sub_bits = 3;
        }

        if (self.sub_bits > 0) {
            self.orig = img.*;
            img.data = img.data[0 .. img.data.len / img.sx * divExpCeil(img.sx, self.sub_bits)];
            img.sx = divExpCeil(img.sx, self.sub_bits);
        }
    }
    pub fn invert(self: *PaletteTf, img: *Image) !void {
        if (self.sub_bits > 0) {
            const orig = self.orig;
            for (0..orig.sy) |y_| {
                const y: u16 = @intCast(orig.sy - y_ - 1);
                for (0..orig.sx) |x_| {
                    const dstx: u16 = @intCast(orig.sx - x_ - 1);
                    const srcx = dstx >> self.sub_bits;
                    const subx: u3 = @truncate(dstx & ((@as(u3, 1) << self.sub_bits) - 1));
                    orig.data[y * orig.sx + dstx] = self.palette[img.get(srcx, y).g >> (subx << (3 - self.sub_bits))];
                }
            }
            img.* = orig;
        } else {
            for (img.data) |*v| {
                v.* = self.palette[v.g];
            }
        }
    }
};

const ColorDecorellationTf = struct {
    img: Image,
    block_size_log: u4,

    pub fn read(self: *ColorDecorellationTf, rd: anytype, alloc: std.mem.Allocator, img: Image) !void {
        const sx = img.sx;
        const sy = img.sy;
        const block_size_log = 2 + try rd.readBitsNoEof(u4, 3);
        self.block_size_log = block_size_log;
        self.img.sx = divExpCeil(sx, block_size_log);
        self.img.sy = divExpCeil(sy, block_size_log);
        self.img.data = try alloc.alloc(Color, @as(u32, divExpCeil(sx, block_size_log)) * divExpCeil(sy, block_size_log));
        try decodeSubImage(alloc, false, rd, self.img);
    }

    pub fn invert(self: *ColorDecorellationTf, img: Image) void {
        for (0..img.sy) |y_| {
            const y: u16 = @intCast(y_);
            for (0..img.sx) |x_| {
                const x: u16 = @intCast(x_);
                const coeff = self.img.get(x >> self.block_size_log, y >> self.block_size_log);
                const dst = x + @as(u32, y) * img.sx;
                img.data[dst] = decorrelate(img.data[dst], coeff);
            }
        }
    }
};

const PredictTf = struct {
    img: Image,
    block_size_log: u4,

    pub fn read(self: *PredictTf, rd: anytype, alloc: std.mem.Allocator, img: Image) !void {
        const sx = img.sx;
        const sy = img.sy;
        const block_size_log = 2 + try rd.readBitsNoEof(u4, 3);
        self.block_size_log = block_size_log;
        self.img.sx = divExpCeil(sx, block_size_log);
        self.img.sy = divExpCeil(sy, block_size_log);
        self.img.data = try alloc.alloc(Color, divExpCeil(sx, block_size_log) * divExpCeil(sy, block_size_log));
        errdefer alloc.free(self.img.data);
        try decodeSubImage(alloc, false, rd, self.img);
    }

    pub fn invert(self: *PredictTf, img: Image) !void {
        const data = img.data;
        data[0].a +%= 255;
        const sx = img.sx;
        for (data[1..sx], data[0 .. sx - 1]) |*dst, prediction| {
            dst.* = addColor(dst.*, prediction);
        }
        for (data[sx..], sx..) |*dst, idx| {
            const x: u16 = @intCast(idx % sx);
            const y: u16 = @intCast(idx / sx);
            const predictor = self.img.get(x >> self.block_size_log, y >> self.block_size_log);
            if (x == 0) {
                dst.* = addColor(dst.*, data[idx - sx]);
            } else {
                const prediction = try predict(
                    predictor.g,
                    data[idx - sx - 1],
                    data[idx - sx],
                    data[idx - sx + 1],
                    data[idx - 1],
                );
                dst.* = addColor(dst.*, prediction);
            }
        }
    }
};

fn readExtraBitsLZ(sym: u16, rd: anytype) !u32 {
    if (sym < 4) return sym + 1;
    const ext: u5 = @intCast((sym - 2) / 2);
    const offs = (2 + @as(u32, sym & 1)) << ext;
    return 1 + offs + try rd.readBitsNoEof(u32, ext);
}

fn mapDistance(dist: u32, stride: u16) u32 {
    if (dist > 120) return dist - 120;

    const offs = offsTable[dist - 1];
    return @intCast(@max(@as(i32, offs[1]) * stride + @as(i32, offs[0]), 1));
}

const offsTable = [120][2]i8{
    .{ 0, 1 },  .{ 1, 0 },  .{ 1, 1 },  .{ -1, 1 }, .{ 0, 2 },  .{ 2, 0 },  .{ 1, 2 },  .{ -1, 2 },
    .{ 2, 1 },  .{ -2, 1 }, .{ 2, 2 },  .{ -2, 2 }, .{ 0, 3 },  .{ 3, 0 },  .{ 1, 3 },  .{ -1, 3 },
    .{ 3, 1 },  .{ -3, 1 }, .{ 2, 3 },  .{ -2, 3 }, .{ 3, 2 },  .{ -3, 2 }, .{ 0, 4 },  .{ 4, 0 },
    .{ 1, 4 },  .{ -1, 4 }, .{ 4, 1 },  .{ -4, 1 }, .{ 3, 3 },  .{ -3, 3 }, .{ 2, 4 },  .{ -2, 4 },
    .{ 4, 2 },  .{ -4, 2 }, .{ 0, 5 },  .{ 3, 4 },  .{ -3, 4 }, .{ 4, 3 },  .{ -4, 3 }, .{ 5, 0 },
    .{ 1, 5 },  .{ -1, 5 }, .{ 5, 1 },  .{ -5, 1 }, .{ 2, 5 },  .{ -2, 5 }, .{ 5, 2 },  .{ -5, 2 },
    .{ 4, 4 },  .{ -4, 4 }, .{ 3, 5 },  .{ -3, 5 }, .{ 5, 3 },  .{ -5, 3 }, .{ 0, 6 },  .{ 6, 0 },
    .{ 1, 6 },  .{ -1, 6 }, .{ 6, 1 },  .{ -6, 1 }, .{ 2, 6 },  .{ -2, 6 }, .{ 6, 2 },  .{ -6, 2 },
    .{ 4, 5 },  .{ -4, 5 }, .{ 5, 4 },  .{ -5, 4 }, .{ 3, 6 },  .{ -3, 6 }, .{ 6, 3 },  .{ -6, 3 },
    .{ 0, 7 },  .{ 7, 0 },  .{ 1, 7 },  .{ -1, 7 }, .{ 5, 5 },  .{ -5, 5 }, .{ 7, 1 },  .{ -7, 1 },
    .{ 4, 6 },  .{ -4, 6 }, .{ 6, 4 },  .{ -6, 4 }, .{ 2, 7 },  .{ -2, 7 }, .{ 7, 2 },  .{ -7, 2 },
    .{ 3, 7 },  .{ -3, 7 }, .{ 7, 3 },  .{ -7, 3 }, .{ 5, 6 },  .{ -5, 6 }, .{ 6, 5 },  .{ -6, 5 },
    .{ 8, 0 },  .{ 4, 7 },  .{ -4, 7 }, .{ 7, 4 },  .{ -7, 4 }, .{ 8, 1 },  .{ 8, 2 },  .{ 6, 6 },
    .{ -6, 6 }, .{ 8, 3 },  .{ 5, 7 },  .{ -5, 7 }, .{ 7, 5 },  .{ -7, 5 }, .{ 8, 4 },  .{ 6, 7 },
    .{ -6, 7 }, .{ 7, 6 },  .{ -7, 6 }, .{ 8, 5 },  .{ 7, 7 },  .{ -7, 7 }, .{ 8, 6 },  .{ 8, 7 },
};

fn ctDelta(t: u8, c: u8) u8 {
    var tmp: i16 = @as(i8, @bitCast(t));
    tmp *= @as(i8, @bitCast(c));
    tmp >>= 5;
    return @bitCast(@as(i8, @truncate(tmp)));
}

fn decorrelate(target: Color, coeffs: Color) Color {
    var x = target;
    x.r +%= ctDelta(x.g, coeffs.b);
    x.b +%= ctDelta(x.g, coeffs.g);
    x.b +%= ctDelta(x.r, coeffs.r);
    return x;
}

fn predict(predictor: u8, tl: Color, t: Color, tr: Color, l: Color) !Color {
    return switch (predictor) {
        else => error.Invalid,
        0 => .{ .a = 255 },
        1 => l,
        2 => t,
        3 => tr,
        4 => tl,
        5 => predictAvg(.{ predictAvg(.{ l, tr }), t }),
        6 => predictAvg(.{ l, tl }),
        7 => predictAvg(.{ l, t }),
        8 => predictAvg(.{ tl, t }),
        9 => predictAvg(.{ t, tr }),
        10 => predictAvg(.{ predictAvg(.{ l, tl }), predictAvg(.{ t, tr }) }),
        11 => predictSel(l, t, tl),
        12 => predictA(.{ l, t, tl }),
        13 => predictB(.{ predictAvg(.{ l, t }), tl }),
    };
}

/// sum of absolute differences
fn sad(a: Color, b: Color) u16 {
    var sum: u16 = 0;
    inline for (components) |comp| {
        sum += @abs(@as(i16, @field(a, comp)) - @field(b, comp));
    }
    return sum;
}

fn predictSel(l: Color, t: Color, tl: Color) Color {
    const p = predictA(.{ l, t, tl });
    return if (sad(p, l) < sad(p, t)) l else t;
}

const predictAvg = componentwise(predictAvg_);
const predictA = componentwise(predictA_);
const predictB = componentwise(predictB_);

fn predictAvg_(a: u8, b: u8) u8 {
    return @intCast((@as(u16, a) + b) / 2);
}
fn predictA_(a: u8, b: u8, c: u8) u8 {
    return @intCast(@max(@min(@as(i16, a) + b - c, 255), 0));
}
fn predictB_(a: u8, b: u8) u8 {
    return @intCast(@max(@min(a + @divTrunc(@as(i16, a) - b, 2), 255), 0));
}

fn componentwise(comptime f: anytype) fn ([@typeInfo(@TypeOf(f)).Fn.params.len]Color) Color {
    const args = @typeInfo(@TypeOf(f)).Fn.params.len;
    return struct {
        fn impl(arr: [args]Color) Color {
            var res: Color = undefined;
            inline for (components) |comp| {
                var tmp: std.meta.ArgsTuple(@TypeOf(f)) = undefined;
                inline for (arr, &tmp) |src, *dst| {
                    dst.* = @field(src, comp);
                }
                @field(res, comp) = @call(.auto, f, tmp);
            }
            return res;
        }
    }.impl;
}

fn addColor(x: Color, y: Color) Color {
    return .{
        .b = x.b +% y.b,
        .g = x.g +% y.g,
        .r = x.r +% y.r,
        .a = x.a +% y.a,
    };
}

fn hash(col: Color, shift: u5) u32 {
    return (@as(u32, @bitCast(col)) *% 0x1e35a7bd) >> shift;
}

pub const Color = packed struct { b: u8 = 0, g: u8 = 0, r: u8 = 0, a: u8 = 0 };
const components = std.meta.fieldNames(Color);

/// x must be bigger than 0
fn divExpCeil(x: u16, y: u4) u16 {
    return ((x - 1) >> y) + 1;
}

const Image = struct {
    data: []Color,
    sx: u16,
    sy: u16,
    pub fn get(self: *const Image, x: u16, y: u16) Color {
        return self.data[x + @as(u32, y) * self.sx];
    }
};

fn readCode(out_lens: []u8, rd: anytype) !void {
    @memset(out_lens, 0);
    if ((try rd.readBitsNoEof(u1, 1)) == 1) {
        const num_syms = 1 + try rd.readBitsNoEof(u2, 1);
        const fs = try rd.readBitsNoEof(u8, if (1 == try rd.readBitsNoEof(u1, 1)) 8 else 1);
        if (fs > out_lens.len) return error.Invalid;
        out_lens[fs] = 1;
        if (num_syms == 2) {
            const ss = try rd.readBitsNoEof(u8, 8);
            if (ss > out_lens.len) return error.Invalid;
            out_lens[ss] = 1;
        }
        return;
    }
    const metalen_cnt = (try rd.readBitsNoEof(u5, 4)) + 4;
    var metalens: [19]u8 = undefined;
    @memset(&metalens, 0);
    const metalen_order = [19]u8{ 17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    for (metalen_order[0..metalen_cnt]) |dst_idx| {
        metalens[dst_idx] = (try rd.readBitsNoEof(u8, 3));
    }
    var tmp1: [19]u16 = undefined;
    var metacode = Code{ .map = &tmp1 };
    try metacode.build(&metalens);
    var sym_count: u32 = undefined;
    if (1 == try rd.readBitsNoEof(u1, 1)) {
        sym_count = 2 + try rd.readBitsNoEof(u32, 2 + 2 * try rd.readBitsNoEof(u5, 3));
        if (sym_count > out_lens.len) return error.Invalid;
    } else {
        sym_count = @intCast(out_lens.len);
    }

    var wdst: u16 = 0;
    var prev_len: u8 = 8;
    while (wdst < out_lens.len and sym_count > 0) : (sym_count -= 1) {
        const tmp = @as(u8, @intCast(try metacode.readSym(rd)));
        switch (tmp) {
            0...15 => |v| {
                out_lens[wdst] = v;
                wdst += 1;
                if (v != 0) prev_len = v;
            },
            16 => {
                const len = 3 + try rd.readBitsNoEof(u3, 2);
                if (out_lens[wdst..].len < len) return error.Invalid;
                @memset(out_lens[wdst..][0..len], prev_len);
                wdst += len;
            },
            17 => {
                const len = 3 + try rd.readBitsNoEof(u4, 3);
                if (out_lens[wdst..].len < len) return error.Invalid;
                @memset(out_lens[wdst..][0..len], 0);
                wdst += len;
            },
            18 => {
                const len = 11 + try rd.readBitsNoEof(u8, 7);
                if (out_lens[wdst..].len < len) return error.Invalid;
                @memset(out_lens[wdst..][0..len], 0);
                wdst += len;
            },
            else => unreachable,
        }
    }
}
