import React from "react";
import {
  AbsoluteFill,
  Easing,
  interpolate,
  spring,
  useCurrentFrame,
  useVideoConfig,
} from "remotion";

const colors = {
  navy: "#0B3A75",
  blue: "#005EA8",
  sky: "#E9F3FF",
  saffron: "#FF9933",
  green: "#138808",
  ink: "#101828",
  muted: "#475467",
  line: "#D0D5DD",
  soft: "#F6F9FC",
  white: "#FFFFFF",
  cream: "#FFF7ED",
};

const ease = Easing.bezier(0.16, 1, 0.3, 1);

const card: React.CSSProperties = {
  background: colors.white,
  border: `1px solid ${colors.line}`,
  borderRadius: 14,
  boxShadow: "0 18px 45px rgba(16, 24, 40, 0.08)",
};

const fade = (
  frame: number,
  start: number,
  end: number,
  from = 0,
  to = 1,
) =>
  interpolate(frame, [start, end], [from, to], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: ease,
  });

const slide = (frame: number, start: number, end: number, px = 40) =>
  interpolate(frame, [start, end], [px, 0], {
    extrapolateLeft: "clamp",
    extrapolateRight: "clamp",
    easing: ease,
  });

const Header = ({ progress }: { progress: number }) => (
  <div
    style={{
      ...card,
      overflow: "hidden",
      transform: `translateY(${(1 - progress) * -42}px)`,
      opacity: progress,
    }}
  >
    <div
      style={{
        height: 46,
        background: colors.navy,
        color: colors.white,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 34px",
        fontSize: 18,
        fontWeight: 650,
      }}
    >
      <div style={{ display: "flex", gap: 14, alignItems: "center" }}>
        <span style={{ color: colors.saffron }}>IN</span>
        <span>An official Government of India digital service</span>
      </div>
      <div style={{ display: "flex", gap: 28 }}>
        <span>English</span>
        <span>Hindi</span>
        <span>A+</span>
        <span>Contrast</span>
      </div>
    </div>
    <div
      style={{
        height: 116,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: "0 34px",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 18 }}>
        <div
          style={{
            height: 62,
            width: 62,
            borderRadius: 32,
            background: colors.cream,
            border: `2px solid ${colors.saffron}`,
            display: "grid",
            placeItems: "center",
            color: colors.navy,
            fontWeight: 800,
            fontSize: 15,
          }}
        >
          BIS
        </div>
        <div>
          <div style={{ fontSize: 29, fontWeight: 800, color: colors.ink }}>
            BIS Standards Recommendation Engine
          </div>
          <div style={{ fontSize: 18, fontWeight: 550, color: colors.muted }}>
            Bureau of Indian Standards lookup for manufacturers
          </div>
        </div>
      </div>
      <div
        style={{
          display: "flex",
          gap: 10,
          fontSize: 18,
          fontWeight: 700,
          color: colors.muted,
        }}
      >
        {["Search", "Catalog", "Evaluation", "API", "Feedback"].map((item) => (
          <div
            key={item}
            style={{
              padding: "10px 16px",
              borderRadius: 8,
              color: item === "Search" ? colors.blue : colors.muted,
              background: item === "Search" ? colors.sky : "transparent",
            }}
          >
            {item}
          </div>
        ))}
      </div>
    </div>
  </div>
);

const SearchCard = ({ typed }: { typed: number }) => {
  const query =
    "33 Grade Ordinary Portland Cement chemical and physical requirements";
  const characters = Math.floor(query.length * typed);

  return (
    <div style={{ ...card, padding: 34, width: 900 }}>
      <div style={{ color: colors.ink, fontWeight: 800, fontSize: 22 }}>
        Describe your product or manufacturing use case
      </div>
      <div
        style={{
          marginTop: 20,
          minHeight: 86,
          border: `2px solid ${colors.blue}`,
          borderRadius: 12,
          background: colors.soft,
          display: "flex",
          alignItems: "center",
          gap: 18,
          padding: "0 24px",
          fontSize: 25,
          fontWeight: 650,
          color: colors.ink,
        }}
      >
        <span
          style={{
            width: 46,
            height: 46,
            borderRadius: 10,
            background: colors.sky,
            display: "grid",
            placeItems: "center",
            color: colors.blue,
            fontWeight: 900,
          }}
        >
          S
        </span>
        <span>{query.slice(0, characters)}</span>
        <span style={{ opacity: typed < 1 ? 1 : 0 }}>|</span>
      </div>
      <div style={{ display: "flex", gap: 14, marginTop: 24 }}>
        <div
          style={{
            background: colors.blue,
            color: colors.white,
            borderRadius: 10,
            padding: "16px 22px",
            fontSize: 22,
            fontWeight: 800,
          }}
        >
          Recommend standards
        </div>
        <div
          style={{
            color: colors.blue,
            border: `2px solid ${colors.blue}`,
            borderRadius: 10,
            padding: "14px 20px",
            fontSize: 22,
            fontWeight: 800,
          }}
        >
          Upload product list
        </div>
      </div>
    </div>
  );
};

const ResultRow = ({
  rank,
  code,
  title,
  active,
}: {
  rank: number;
  code: string;
  title: string;
  active?: boolean;
}) => (
  <div
    style={{
      ...card,
      borderColor: active ? "#8ED6A6" : colors.line,
      background: active ? "#F0FDF4" : colors.white,
      padding: "18px 22px",
      display: "flex",
      alignItems: "center",
      gap: 20,
      height: 100,
    }}
  >
    <div
      style={{
        width: 56,
        height: 56,
        borderRadius: 28,
        display: "grid",
        placeItems: "center",
        background: active ? colors.green : colors.sky,
        color: active ? colors.white : colors.blue,
        fontWeight: 900,
        fontSize: 19,
      }}
    >
      #{rank}
    </div>
    <div style={{ flex: 1 }}>
      <div style={{ fontSize: 24, fontWeight: 850, color: colors.ink }}>
        {code}
      </div>
      <div style={{ marginTop: 3, fontSize: 17, color: colors.muted }}>
        {title}
      </div>
    </div>
    <div
      style={{
        borderRadius: 999,
        background: active ? "#DCFCE7" : colors.sky,
        color: active ? colors.green : colors.blue,
        fontSize: 16,
        fontWeight: 800,
        padding: "10px 14px",
      }}
    >
      {active ? "Matched expected" : "Related"}
    </div>
  </div>
);

const Metric = ({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) => (
  <div style={{ ...card, width: 240, padding: 22 }}>
    <div style={{ color: colors.muted, fontSize: 17, fontWeight: 750 }}>
      {label}
    </div>
    <div style={{ color, fontSize: 44, fontWeight: 900, marginTop: 8 }}>
      {value}
    </div>
  </div>
);

export const BisPlatformWalkthrough = () => {
  const frame = useCurrentFrame();
  const { fps } = useVideoConfig();

  const headerProgress = fade(frame, 0, 1.2 * fps);
  const heroProgress = spring({ frame: frame - 18, fps, config: { damping: 18 } });
  const typed = fade(frame, 2.8 * fps, 5.6 * fps);
  const resultProgress = fade(frame, 5.8 * fps, 7.4 * fps);
  const metricsProgress = fade(frame, 8.4 * fps, 10 * fps);
  const closingProgress = fade(frame, 10.4 * fps, 11.4 * fps);

  return (
    <AbsoluteFill
      style={{
        background: colors.soft,
        fontFamily: "Noto Sans, Arial, sans-serif",
        color: colors.ink,
        padding: 72,
      }}
    >
      <Header progress={headerProgress} />

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 420px",
          gap: 34,
          marginTop: 42,
          transform: `translateY(${(1 - heroProgress) * 42}px)`,
          opacity: heroProgress,
        }}
      >
        <div>
          <div
            style={{
              color: colors.blue,
              fontWeight: 900,
              fontSize: 20,
              letterSpacing: 1.8,
            }}
          >
            BIS DIGITAL SERVICE
          </div>
          <div
            style={{
              fontSize: 58,
              lineHeight: 1.05,
              fontWeight: 900,
              maxWidth: 920,
              marginTop: 16,
            }}
          >
            Find the applicable Indian Standard from a product query
          </div>
          <div
            style={{
              color: colors.muted,
              fontSize: 24,
              lineHeight: 1.45,
              maxWidth: 900,
              marginTop: 22,
            }}
          >
            Search manufacturing descriptions, abbreviations like OPC/PPC/CMU,
            or explicit IS codes. The platform returns five standards with
            transparent ranking evidence.
          </div>

          <div
            style={{
              marginTop: 36,
              transform: `translateY(${slide(frame, 2 * fps, 3 * fps, 28)}px)`,
            }}
          >
            <SearchCard typed={typed} />
          </div>
        </div>

        <div style={{ display: "grid", gap: 18 }}>
          <div style={{ ...card, padding: 24 }}>
            <div style={{ fontSize: 26, fontWeight: 900 }}>
              GIGW service readiness
            </div>
            {[
              "Official .gov.in deployment cue",
              "Bilingual and accessible controls",
              "Visible feedback mechanism",
              "Content owner and update status",
            ].map((item) => (
              <div
                key={item}
                style={{
                  display: "flex",
                  gap: 12,
                  alignItems: "center",
                  marginTop: 17,
                  fontSize: 19,
                  color: colors.muted,
                  fontWeight: 650,
                }}
              >
                <span style={{ color: colors.green, fontWeight: 900 }}>OK</span>
                {item}
              </div>
            ))}
          </div>
          <div style={{ ...card, padding: 24, background: colors.navy }}>
            <div style={{ color: colors.white, fontSize: 25, fontWeight: 900 }}>
              Accessibility tools
            </div>
            <div style={{ color: "#DDEBFF", fontSize: 18, lineHeight: 1.45 }}>
              Text size, contrast, language, keyboard help and service feedback
              stay visible throughout the workflow.
            </div>
          </div>
        </div>
      </div>

      <div
        style={{
          position: "absolute",
          left: 72,
          right: 72,
          bottom: 68,
          display: "grid",
          gridTemplateColumns: "1fr 360px",
          gap: 26,
          opacity: resultProgress,
          transform: `translateY(${(1 - resultProgress) * 52}px)`,
        }}
      >
        <div style={{ display: "grid", gap: 14 }}>
          <ResultRow
            rank={1}
            code="IS 269: 1989"
            title="Specification for 33 grade ordinary Portland cement"
            active
          />
          <ResultRow
            rank={2}
            code="IS 4985: 1988"
            title="Unplasticized PVC pipes for potable water supplies"
          />
          <ResultRow
            rank={3}
            code="IS 455: 1989"
            title="Portland slag cement"
          />
        </div>
        <div style={{ display: "grid", gap: 14, opacity: metricsProgress }}>
          <Metric label="Hit Rate @3" value="100%" color={colors.green} />
          <Metric label="MRR @5" value="1.000" color={colors.blue} />
          <Metric label="Latency" value="0.03s" color={colors.saffron} />
        </div>
      </div>

      <div
        style={{
          position: "absolute",
          inset: 0,
          background: `rgba(11, 58, 117, ${closingProgress * 0.96})`,
          display: "grid",
          placeItems: "center",
          opacity: closingProgress,
        }}
      >
        <div style={{ textAlign: "center", color: colors.white }}>
          <div style={{ fontSize: 70, fontWeight: 950 }}>
            Submission-ready public service UI
          </div>
          <div style={{ fontSize: 30, marginTop: 20, color: "#DDEBFF" }}>
            UX4G-informed design, deterministic retrieval, evaluator-compliant
            output.
          </div>
        </div>
      </div>
    </AbsoluteFill>
  );
};
