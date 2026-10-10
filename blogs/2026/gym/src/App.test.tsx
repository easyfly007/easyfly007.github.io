// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import App from './App';

function click(name: string | RegExp) {
    fireEvent.click(screen.getByRole('button', { name }));
}

describe('App flow', () => {
    beforeEach(() => {
        window.localStorage.clear();
        vi.useFakeTimers();
        vi.spyOn(window, 'prompt').mockImplementation((_m, def) => def ?? '');
        vi.spyOn(window, 'confirm').mockReturnValue(true);
    });

    afterEach(() => {
        cleanup();
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it('generates, saves, and plays a workout', () => {
        render(<App />);
        click('哑铃');
        click('生成训练');

        expect(screen.getByText('热身')).toBeTruthy();
        expect(screen.getByText(/^主训练/)).toBeTruthy();
        expect(screen.getByText('放松')).toBeTruthy();

        click('保存为模板');
        expect(JSON.parse(window.localStorage.getItem('gym.saved.v1')!)).toHaveLength(1);

        click('▶ 开始跟练');
        expect(screen.getByText('准备')).toBeTruthy();
        expect(screen.getByText('10')).toBeTruthy();

        act(() => {
            vi.advanceTimersByTime(3000);
        });
        expect(screen.getByText('7')).toBeTruthy();

        click('跳过');
        act(() => {
            vi.advanceTimersByTime(200);
        });
        expect(document.querySelector('.player.work')).toBeTruthy();

        click('暂停');
        const before = document.querySelector('.countdown')!.textContent;
        act(() => {
            vi.advanceTimersByTime(5000);
        });
        expect(document.querySelector('.countdown')!.textContent).toBe(before);
        click('继续');

        click('结束');
        click('修改条件');
        const saved = screen.getByText('我的模板').closest('section')!;
        expect(within(saved).getByText(/^室内-瑜伽垫\+哑铃-30分钟/)).toBeTruthy();
    });

    it('restores the last conditions on reload', () => {
        const first = render(<App />);
        click('杠铃');
        click('生成训练');
        first.unmount();

        render(<App />);
        expect(screen.getByRole('button', { name: '杠铃' }).getAttribute('aria-pressed')).toBe('true');
    });
});
