import { Bot } from 'lucide-react';
import { motion } from 'motion/react';

interface BotIconProps {
onBotClick: () => void;
isChatOpen?: boolean;
}

export function BotIcon({ onBotClick, isChatOpen = false }: BotIconProps) {
return (
<div className="absolute top-32 right-9 z-20">
    {/* Bot Icon - spins like a coin every 20 seconds */}
    <motion.div
    layout
    initial={false}
    animate={
        isChatOpen
        ? {
            width: 0,
            height: 0,
            opacity: 0,
            scale: 0,
            }
        : {
            width: 80,
            height: 80,
            opacity: 1,
            scale: 1,
            }
    }
    transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
    className="relative overflow-hidden"
    style={{ transformOrigin: 'center' }}
    >
    {!isChatOpen && (
        <motion.button
        onClick={onBotClick}
        className="w-9 h-9 bg-blue-600 text-white rounded-full hover:bg-blue-700 transition-colors flex items-center justify-center shadow-lg border border-gray-200"
        animate={{
            rotateY: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 360, 0],
        }}
        transition={{
            duration: 20,
            repeat: Infinity,
            ease: "linear",
            times: [0, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1],
        }}
        >
        <Bot className="w-6 h-6" />
        </motion.button>
    )}
    </motion.div>
</div>
);
}
