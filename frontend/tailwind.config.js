/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      colors: {
        ink: {
          950: "#080808",
          900: "#111111",
          800: "#1f1f1f",
          700: "#333333",
          600: "#555555",
          400: "#888888",
          200: "#cccccc",
          100: "#e8e8e8",
          50:  "#f5f5f5",
        },
      },
      keyframes: {
        "fade-in": {
          from: { opacity: "0", transform: "translateY(6px)" },
          to:   { opacity: "1", transform: "translateY(0)" },
        },
        "pulse-dot": {
          "0%, 80%, 100%": { transform: "scale(0.8)", opacity: "0.4" },
          "40%":           { transform: "scale(1)",   opacity: "1"   },
        },
      },
      animation: {
        "fade-in": "fade-in 0.25s ease-out both",
        "pulse-dot": "pulse-dot 1.2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
